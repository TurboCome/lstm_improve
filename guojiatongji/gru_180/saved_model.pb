??8
??
B
AddV2
x"T
y"T
z"T"
Ttype:
2	??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
.
Identity

input"T
output"T"	
Ttype
:
Less
x"T
y"T
z
"
Ttype:
2	
?
	MLCMatMul
a"T
b"T

unique_key"T*num_args
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
2"
num_argsint ("

input_rankint(0
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
=
Mul
x"T
y"T
z"T"
Ttype:
2	?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
?
PartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
[
Split
	split_dim

value"T
output"T*	num_split"
	num_splitint(0"	
Ttype
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
?
StatelessWhile

input2T
output2T"
T
list(type)("
condfunc"
bodyfunc" 
output_shapeslist(shape)
 "
parallel_iterationsint

@
StaticRegexFullMatch	
input

output
"
patternstring
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
;
Sub
x"T
y"T
z"T"
Ttype:
2	
-
Tanh
x"T
y"T"
Ttype:

2
?
TensorListFromTensor
tensor"element_dtype
element_shape"
shape_type
output_handle"
element_dtypetype"

shape_typetype:
2	
?
TensorListReserve
element_shape"
shape_type
num_elements

handle"
element_dtypetype"

shape_typetype:
2	
?
TensorListStack
input_handle
element_shape
tensor"element_dtype"
element_dtypetype" 
num_elementsint?????????
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
P
Unpack

value"T
output"T*num"
numint("	
Ttype"
axisint 
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*	2.4.0-rc02&tf_macos-v0.1-alpha2-AS-67-gf3595294ab8??6
y
dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*
shared_namedense_2/kernel
r
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel*
_output_shapes
:	?*
dtype0
q
dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_2/bias
j
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
_output_shapes	
:?*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
?
gru/gru_cell/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:Z*$
shared_namegru/gru_cell/kernel
{
'gru/gru_cell/kernel/Read/ReadVariableOpReadVariableOpgru/gru_cell/kernel*
_output_shapes

:Z*
dtype0
?
gru/gru_cell/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:Z*.
shared_namegru/gru_cell/recurrent_kernel
?
1gru/gru_cell/recurrent_kernel/Read/ReadVariableOpReadVariableOpgru/gru_cell/recurrent_kernel*
_output_shapes

:Z*
dtype0
~
gru/gru_cell/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:Z*"
shared_namegru/gru_cell/bias
w
%gru/gru_cell/bias/Read/ReadVariableOpReadVariableOpgru/gru_cell/bias*
_output_shapes

:Z*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_2
[
total_2/Read/ReadVariableOpReadVariableOptotal_2*
_output_shapes
: *
dtype0
b
count_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_2
[
count_2/Read/ReadVariableOpReadVariableOpcount_2*
_output_shapes
: *
dtype0
?
Adam/dense_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*&
shared_nameAdam/dense_2/kernel/m
?
)Adam/dense_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_2/kernel/m*
_output_shapes
:	?*
dtype0

Adam/dense_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*$
shared_nameAdam/dense_2/bias/m
x
'Adam/dense_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_2/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/gru/gru_cell/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:Z*+
shared_nameAdam/gru/gru_cell/kernel/m
?
.Adam/gru/gru_cell/kernel/m/Read/ReadVariableOpReadVariableOpAdam/gru/gru_cell/kernel/m*
_output_shapes

:Z*
dtype0
?
$Adam/gru/gru_cell/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:Z*5
shared_name&$Adam/gru/gru_cell/recurrent_kernel/m
?
8Adam/gru/gru_cell/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp$Adam/gru/gru_cell/recurrent_kernel/m*
_output_shapes

:Z*
dtype0
?
Adam/gru/gru_cell/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:Z*)
shared_nameAdam/gru/gru_cell/bias/m
?
,Adam/gru/gru_cell/bias/m/Read/ReadVariableOpReadVariableOpAdam/gru/gru_cell/bias/m*
_output_shapes

:Z*
dtype0
?
Adam/dense_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*&
shared_nameAdam/dense_2/kernel/v
?
)Adam/dense_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_2/kernel/v*
_output_shapes
:	?*
dtype0

Adam/dense_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*$
shared_nameAdam/dense_2/bias/v
x
'Adam/dense_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_2/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/gru/gru_cell/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:Z*+
shared_nameAdam/gru/gru_cell/kernel/v
?
.Adam/gru/gru_cell/kernel/v/Read/ReadVariableOpReadVariableOpAdam/gru/gru_cell/kernel/v*
_output_shapes

:Z*
dtype0
?
$Adam/gru/gru_cell/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:Z*5
shared_name&$Adam/gru/gru_cell/recurrent_kernel/v
?
8Adam/gru/gru_cell/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp$Adam/gru/gru_cell/recurrent_kernel/v*
_output_shapes

:Z*
dtype0
?
Adam/gru/gru_cell/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:Z*)
shared_nameAdam/gru/gru_cell/bias/v
?
,Adam/gru/gru_cell/bias/v/Read/ReadVariableOpReadVariableOpAdam/gru/gru_cell/bias/v*
_output_shapes

:Z*
dtype0

NoOpNoOp
?$
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?#
value?#B?# B?#
?
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api

signatures
l
	cell


state_spec
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
?
iter

beta_1

beta_2
	decay
learning_ratemGmHmImJmKvLvMvNvOvP
#
0
1
2
3
4
#
0
1
2
3
4
 
?
layer_metrics
	variables
trainable_variables
metrics
regularization_losses

layers
 non_trainable_variables
!layer_regularization_losses
 
~

kernel
recurrent_kernel
bias
"	variables
#trainable_variables
$regularization_losses
%	keras_api
 

0
1
2

0
1
2
 
?
&layer_metrics
	variables

'states
trainable_variables
(metrics
regularization_losses

)layers
*non_trainable_variables
+layer_regularization_losses
ZX
VARIABLE_VALUEdense_2/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_2/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
,layer_metrics
	variables
trainable_variables
-metrics
regularization_losses

.layers
/non_trainable_variables
0layer_regularization_losses
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEgru/gru_cell/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEgru/gru_cell/recurrent_kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEgru/gru_cell/bias&variables/2/.ATTRIBUTES/VARIABLE_VALUE
 

10
21
32

0
1
 
 

0
1
2

0
1
2
 
?
4layer_metrics
"	variables
#trainable_variables
5metrics
$regularization_losses

6layers
7non_trainable_variables
8layer_regularization_losses
 
 
 

	0
 
 
 
 
 
 
 
4
	9total
	:count
;	variables
<	keras_api
D
	=total
	>count
?
_fn_kwargs
@	variables
A	keras_api
D
	Btotal
	Ccount
D
_fn_kwargs
E	variables
F	keras_api
 
 
 
 
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

90
:1

;	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

=0
>1

@	variables
QO
VARIABLE_VALUEtotal_24keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_24keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUE
 

B0
C1

E	variables
}{
VARIABLE_VALUEAdam/dense_2/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_2/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUEAdam/gru/gru_cell/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE$Adam/gru/gru_cell/recurrent_kernel/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/gru/gru_cell/bias/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_2/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_2/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUEAdam/gru/gru_cell/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE$Adam/gru/gru_cell/recurrent_kernel/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/gru/gru_cell/bias/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_gru_inputPlaceholder*,
_output_shapes
:??????????*
dtype0*!
shape:??????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_gru_inputgru/gru_cell/kernelgru/gru_cell/recurrent_kernelgru/gru_cell/biasdense_2/kerneldense_2/bias*
Tin

2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8? *,
f'R%
#__inference_signature_wrapper_17297
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename"dense_2/kernel/Read/ReadVariableOp dense_2/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp'gru/gru_cell/kernel/Read/ReadVariableOp1gru/gru_cell/recurrent_kernel/Read/ReadVariableOp%gru/gru_cell/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal_2/Read/ReadVariableOpcount_2/Read/ReadVariableOp)Adam/dense_2/kernel/m/Read/ReadVariableOp'Adam/dense_2/bias/m/Read/ReadVariableOp.Adam/gru/gru_cell/kernel/m/Read/ReadVariableOp8Adam/gru/gru_cell/recurrent_kernel/m/Read/ReadVariableOp,Adam/gru/gru_cell/bias/m/Read/ReadVariableOp)Adam/dense_2/kernel/v/Read/ReadVariableOp'Adam/dense_2/bias/v/Read/ReadVariableOp.Adam/gru/gru_cell/kernel/v/Read/ReadVariableOp8Adam/gru/gru_cell/recurrent_kernel/v/Read/ReadVariableOp,Adam/gru/gru_cell/bias/v/Read/ReadVariableOpConst*'
Tin 
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *'
f"R 
__inference__traced_save_20320
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_2/kerneldense_2/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rategru/gru_cell/kernelgru/gru_cell/recurrent_kernelgru/gru_cell/biastotalcounttotal_1count_1total_2count_2Adam/dense_2/kernel/mAdam/dense_2/bias/mAdam/gru/gru_cell/kernel/m$Adam/gru/gru_cell/recurrent_kernel/mAdam/gru/gru_cell/bias/mAdam/dense_2/kernel/vAdam/dense_2/bias/vAdam/gru/gru_cell/kernel/v$Adam/gru/gru_cell/recurrent_kernel/vAdam/gru/gru_cell/bias/v*&
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? **
f%R#
!__inference__traced_restore_20408??6
??
?
8__inference___backward_gpu_gru_with_fallback_19488_19706
placeholder
placeholder_1
placeholder_2
placeholder_3/
+gradients_strided_slice_grad_shape_cudnnrnnB
>gradients_strided_slice_grad_mlcstridedslicegrad_strided_sliceA
=gradients_transpose_7_grad_invertpermutation_transpose_7_perm;
7gradients_transpose_7_grad_mlctransposegrad_transpose_7)
%gradients_squeeze_grad_shape_cudnnrnn!
gradients_zeros_like_cudnnrnn#
gradients_zeros_like_1_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims=
9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c3
/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h)
%gradients_concat_grad_add_concat_axis5
1gradients_reshape_1_grad_mlcreshapegrad_reshape_15
1gradients_reshape_2_grad_mlcreshapegrad_reshape_25
1gradients_reshape_3_grad_mlcreshapegrad_reshape_35
1gradients_reshape_4_grad_mlcreshapegrad_reshape_45
1gradients_reshape_5_grad_mlcreshapegrad_reshape_55
1gradients_reshape_6_grad_mlcreshapegrad_reshape_65
1gradients_reshape_7_grad_mlcreshapegrad_reshape_75
1gradients_reshape_8_grad_mlcreshapegrad_reshape_85
1gradients_reshape_9_grad_mlcreshapegrad_reshape_97
3gradients_reshape_10_grad_mlcreshapegrad_reshape_107
3gradients_reshape_11_grad_mlcreshapegrad_reshape_117
3gradients_reshape_12_grad_mlcreshapegrad_reshape_12A
=gradients_transpose_1_grad_invertpermutation_transpose_1_perm;
7gradients_transpose_1_grad_mlctransposegrad_transpose_1A
=gradients_transpose_2_grad_invertpermutation_transpose_2_perm;
7gradients_transpose_2_grad_mlctransposegrad_transpose_2A
=gradients_transpose_3_grad_invertpermutation_transpose_3_perm;
7gradients_transpose_3_grad_mlctransposegrad_transpose_3A
=gradients_transpose_4_grad_invertpermutation_transpose_4_perm;
7gradients_transpose_4_grad_mlctransposegrad_transpose_4A
=gradients_transpose_5_grad_invertpermutation_transpose_5_perm;
7gradients_transpose_5_grad_mlctransposegrad_transpose_5A
=gradients_transpose_6_grad_invertpermutation_transpose_6_perm;
7gradients_transpose_6_grad_mlctransposegrad_transpose_63
/gradients_split_2_grad_concat_split_2_split_dim5
1gradients_split_grad_mlcsplitgrad_split_split_dim+
'gradients_split_grad_mlcsplitgrad_split9
5gradients_split_1_grad_mlcsplitgrad_split_1_split_dim/
+gradients_split_1_grad_mlcsplitgrad_split_11
-gradients_reshape_grad_mlcreshapegrad_reshape
identity

identity_1

identity_2

identity_3

identity_4?u
gradients/grad_ys_0Identityplaceholder*
T0*'
_output_shapes
:?????????2
gradients/grad_ys_0|
gradients/grad_ys_1Identityplaceholder_1*
T0*,
_output_shapes
:??????????2
gradients/grad_ys_1w
gradients/grad_ys_2Identityplaceholder_2*
T0*'
_output_shapes
:?????????2
gradients/grad_ys_2f
gradients/grad_ys_3Identityplaceholder_3*
T0*
_output_shapes
: 2
gradients/grad_ys_3?
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:2$
"gradients/strided_slice_grad/Shape?
6gradients/strided_slice_grad/MLCStridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*
valueB:
?????????28
6gradients/strided_slice_grad/MLCStridedSliceGrad/begin?
4gradients/strided_slice_grad/MLCStridedSliceGrad/endConst*
_output_shapes
:*
dtype0*
valueB: 26
4gradients/strided_slice_grad/MLCStridedSliceGrad/end?
8gradients/strided_slice_grad/MLCStridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*
valueB:2:
8gradients/strided_slice_grad/MLCStridedSliceGrad/strides?
0gradients/strided_slice_grad/MLCStridedSliceGradMLCStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0?gradients/strided_slice_grad/MLCStridedSliceGrad/begin:output:0=gradients/strided_slice_grad/MLCStridedSliceGrad/end:output:0Agradients/strided_slice_grad/MLCStridedSliceGrad/strides:output:0gradients/grad_ys_0:output:0>gradients_strided_slice_grad_mlcstridedslicegrad_strided_slice*
Index0*
T0*,
_output_shapes
:??????????*
shrink_axis_mask22
0gradients/strided_slice_grad/MLCStridedSliceGrad?
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:2.
,gradients/transpose_7_grad/InvertPermutation?
+gradients/transpose_7_grad/MLCTransposeGradMLCTransposeGradgradients/grad_ys_1:output:00gradients/transpose_7_grad/InvertPermutation:y:07gradients_transpose_7_grad_mlctransposegrad_transpose_7*
T0*,
_output_shapes
:??????????2-
+gradients/transpose_7_grad/MLCTransposeGrad?
gradients/Squeeze_grad/ShapeShape%gradients_squeeze_grad_shape_cudnnrnn*
T0*
_output_shapes
:2
gradients/Squeeze_grad/Shape?
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*+
_output_shapes
:?????????2 
gradients/Squeeze_grad/Reshape?
gradients/AddNAddN9gradients/strided_slice_grad/MLCStridedSliceGrad:output:0/gradients/transpose_7_grad/MLCTransposeGrad:y:0*
N*
T0*C
_class9
75loc:@gradients/strided_slice_grad/MLCStridedSliceGrad*,
_output_shapes
:??????????2
gradients/AddNy
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
: 2
gradients/zeros_like?
gradients/zeros_like_1	ZerosLikegradients_zeros_like_1_cudnnrnn*
T0*
_output_shapes
:2
gradients/zeros_like_1?
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat+gradients_strided_slice_grad_shape_cudnnrnn%gradients_squeeze_grad_shape_cudnnrnngradients_zeros_like_cudnnrnngradients/AddN:sum:0'gradients/Squeeze_grad/Reshape:output:0gradients/zeros_like:y:0gradients_zeros_like_1_cudnnrnn*
T0*L
_output_shapes:
8:??????????:?????????: :?*
rnn_modegru2*
(gradients/CudnnRNN_grad/CudnnRNNBackprop?
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:2,
*gradients/transpose_grad/InvertPermutation?
)gradients/transpose_grad/MLCTransposeGradMLCTransposeGrad9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:02gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose*
T0*,
_output_shapes
:??????????2+
)gradients/transpose_grad/MLCTransposeGrad?
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
:2!
gradients/ExpandDims_grad/Shape?
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*'
_output_shapes
:?????????2#
!gradients/ExpandDims_grad/Reshape?
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:?2
gradients/concat_grad/Shape|
gradients/concat_grad/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_grad/add/y?
gradients/concat_grad/addAddV2%gradients_concat_grad_add_concat_axis$gradients/concat_grad/add/y:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/add?
)gradients/concat_grad/strided_slice/stackPack%gradients_concat_grad_add_concat_axis*
N*
T0*
_output_shapes
:2+
)gradients/concat_grad/strided_slice/stack?
+gradients/concat_grad/strided_slice/stack_1Packgradients/concat_grad/add:z:0*
N*
T0*
_output_shapes
:2-
+gradients/concat_grad/strided_slice/stack_1?
+gradients/concat_grad/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+gradients/concat_grad/strided_slice/stack_2?
#gradients/concat_grad/strided_sliceStridedSlice$gradients/concat_grad/Shape:output:02gradients/concat_grad/strided_slice/stack:output:04gradients/concat_grad/strided_slice/stack_1:output:04gradients/concat_grad/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#gradients/concat_grad/strided_slice?
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:?2
gradients/concat_grad/Shape_1?
gradients/concat_grad/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_grad/add_1/y?
gradients/concat_grad/add_1AddV2%gradients_concat_grad_add_concat_axis&gradients/concat_grad/add_1/y:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/add_1?
+gradients/concat_grad/strided_slice_1/stackPack%gradients_concat_grad_add_concat_axis*
N*
T0*
_output_shapes
:2-
+gradients/concat_grad/strided_slice_1/stack?
-gradients/concat_grad/strided_slice_1/stack_1Packgradients/concat_grad/add_1:z:0*
N*
T0*
_output_shapes
:2/
-gradients/concat_grad/strided_slice_1/stack_1?
-gradients/concat_grad/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-gradients/concat_grad/strided_slice_1/stack_2?
%gradients/concat_grad/strided_slice_1StridedSlice&gradients/concat_grad/Shape_1:output:04gradients/concat_grad/strided_slice_1/stack:output:06gradients/concat_grad/strided_slice_1/stack_1:output:06gradients/concat_grad/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%gradients/concat_grad/strided_slice_1?
gradients/concat_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:?2
gradients/concat_grad/Shape_2?
gradients/concat_grad/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_grad/add_2/y?
gradients/concat_grad/add_2AddV2%gradients_concat_grad_add_concat_axis&gradients/concat_grad/add_2/y:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/add_2?
+gradients/concat_grad/strided_slice_2/stackPack%gradients_concat_grad_add_concat_axis*
N*
T0*
_output_shapes
:2-
+gradients/concat_grad/strided_slice_2/stack?
-gradients/concat_grad/strided_slice_2/stack_1Packgradients/concat_grad/add_2:z:0*
N*
T0*
_output_shapes
:2/
-gradients/concat_grad/strided_slice_2/stack_1?
-gradients/concat_grad/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-gradients/concat_grad/strided_slice_2/stack_2?
%gradients/concat_grad/strided_slice_2StridedSlice&gradients/concat_grad/Shape_2:output:04gradients/concat_grad/strided_slice_2/stack:output:06gradients/concat_grad/strided_slice_2/stack_1:output:06gradients/concat_grad/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%gradients/concat_grad/strided_slice_2?
gradients/concat_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:?2
gradients/concat_grad/Shape_3?
gradients/concat_grad/add_3/yConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_grad/add_3/y?
gradients/concat_grad/add_3AddV2%gradients_concat_grad_add_concat_axis&gradients/concat_grad/add_3/y:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/add_3?
+gradients/concat_grad/strided_slice_3/stackPack%gradients_concat_grad_add_concat_axis*
N*
T0*
_output_shapes
:2-
+gradients/concat_grad/strided_slice_3/stack?
-gradients/concat_grad/strided_slice_3/stack_1Packgradients/concat_grad/add_3:z:0*
N*
T0*
_output_shapes
:2/
-gradients/concat_grad/strided_slice_3/stack_1?
-gradients/concat_grad/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-gradients/concat_grad/strided_slice_3/stack_2?
%gradients/concat_grad/strided_slice_3StridedSlice&gradients/concat_grad/Shape_3:output:04gradients/concat_grad/strided_slice_3/stack:output:06gradients/concat_grad/strided_slice_3/stack_1:output:06gradients/concat_grad/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%gradients/concat_grad/strided_slice_3?
gradients/concat_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:?2
gradients/concat_grad/Shape_4?
gradients/concat_grad/add_4/yConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_grad/add_4/y?
gradients/concat_grad/add_4AddV2%gradients_concat_grad_add_concat_axis&gradients/concat_grad/add_4/y:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/add_4?
+gradients/concat_grad/strided_slice_4/stackPack%gradients_concat_grad_add_concat_axis*
N*
T0*
_output_shapes
:2-
+gradients/concat_grad/strided_slice_4/stack?
-gradients/concat_grad/strided_slice_4/stack_1Packgradients/concat_grad/add_4:z:0*
N*
T0*
_output_shapes
:2/
-gradients/concat_grad/strided_slice_4/stack_1?
-gradients/concat_grad/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-gradients/concat_grad/strided_slice_4/stack_2?
%gradients/concat_grad/strided_slice_4StridedSlice&gradients/concat_grad/Shape_4:output:04gradients/concat_grad/strided_slice_4/stack:output:06gradients/concat_grad/strided_slice_4/stack_1:output:06gradients/concat_grad/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%gradients/concat_grad/strided_slice_4?
gradients/concat_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:?2
gradients/concat_grad/Shape_5?
gradients/concat_grad/add_5/yConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_grad/add_5/y?
gradients/concat_grad/add_5AddV2%gradients_concat_grad_add_concat_axis&gradients/concat_grad/add_5/y:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/add_5?
+gradients/concat_grad/strided_slice_5/stackPack%gradients_concat_grad_add_concat_axis*
N*
T0*
_output_shapes
:2-
+gradients/concat_grad/strided_slice_5/stack?
-gradients/concat_grad/strided_slice_5/stack_1Packgradients/concat_grad/add_5:z:0*
N*
T0*
_output_shapes
:2/
-gradients/concat_grad/strided_slice_5/stack_1?
-gradients/concat_grad/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-gradients/concat_grad/strided_slice_5/stack_2?
%gradients/concat_grad/strided_slice_5StridedSlice&gradients/concat_grad/Shape_5:output:04gradients/concat_grad/strided_slice_5/stack:output:06gradients/concat_grad/strided_slice_5/stack_1:output:06gradients/concat_grad/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%gradients/concat_grad/strided_slice_5?
gradients/concat_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape_6?
gradients/concat_grad/add_6/yConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_grad/add_6/y?
gradients/concat_grad/add_6AddV2%gradients_concat_grad_add_concat_axis&gradients/concat_grad/add_6/y:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/add_6?
+gradients/concat_grad/strided_slice_6/stackPack%gradients_concat_grad_add_concat_axis*
N*
T0*
_output_shapes
:2-
+gradients/concat_grad/strided_slice_6/stack?
-gradients/concat_grad/strided_slice_6/stack_1Packgradients/concat_grad/add_6:z:0*
N*
T0*
_output_shapes
:2/
-gradients/concat_grad/strided_slice_6/stack_1?
-gradients/concat_grad/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-gradients/concat_grad/strided_slice_6/stack_2?
%gradients/concat_grad/strided_slice_6StridedSlice&gradients/concat_grad/Shape_6:output:04gradients/concat_grad/strided_slice_6/stack:output:06gradients/concat_grad/strided_slice_6/stack_1:output:06gradients/concat_grad/strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%gradients/concat_grad/strided_slice_6?
gradients/concat_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape_7?
gradients/concat_grad/add_7/yConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_grad/add_7/y?
gradients/concat_grad/add_7AddV2%gradients_concat_grad_add_concat_axis&gradients/concat_grad/add_7/y:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/add_7?
+gradients/concat_grad/strided_slice_7/stackPack%gradients_concat_grad_add_concat_axis*
N*
T0*
_output_shapes
:2-
+gradients/concat_grad/strided_slice_7/stack?
-gradients/concat_grad/strided_slice_7/stack_1Packgradients/concat_grad/add_7:z:0*
N*
T0*
_output_shapes
:2/
-gradients/concat_grad/strided_slice_7/stack_1?
-gradients/concat_grad/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-gradients/concat_grad/strided_slice_7/stack_2?
%gradients/concat_grad/strided_slice_7StridedSlice&gradients/concat_grad/Shape_7:output:04gradients/concat_grad/strided_slice_7/stack:output:06gradients/concat_grad/strided_slice_7/stack_1:output:06gradients/concat_grad/strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%gradients/concat_grad/strided_slice_7?
gradients/concat_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape_8?
gradients/concat_grad/add_8/yConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_grad/add_8/y?
gradients/concat_grad/add_8AddV2%gradients_concat_grad_add_concat_axis&gradients/concat_grad/add_8/y:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/add_8?
+gradients/concat_grad/strided_slice_8/stackPack%gradients_concat_grad_add_concat_axis*
N*
T0*
_output_shapes
:2-
+gradients/concat_grad/strided_slice_8/stack?
-gradients/concat_grad/strided_slice_8/stack_1Packgradients/concat_grad/add_8:z:0*
N*
T0*
_output_shapes
:2/
-gradients/concat_grad/strided_slice_8/stack_1?
-gradients/concat_grad/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-gradients/concat_grad/strided_slice_8/stack_2?
%gradients/concat_grad/strided_slice_8StridedSlice&gradients/concat_grad/Shape_8:output:04gradients/concat_grad/strided_slice_8/stack:output:06gradients/concat_grad/strided_slice_8/stack_1:output:06gradients/concat_grad/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%gradients/concat_grad/strided_slice_8?
gradients/concat_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape_9?
gradients/concat_grad/add_9/yConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_grad/add_9/y?
gradients/concat_grad/add_9AddV2%gradients_concat_grad_add_concat_axis&gradients/concat_grad/add_9/y:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/add_9?
+gradients/concat_grad/strided_slice_9/stackPack%gradients_concat_grad_add_concat_axis*
N*
T0*
_output_shapes
:2-
+gradients/concat_grad/strided_slice_9/stack?
-gradients/concat_grad/strided_slice_9/stack_1Packgradients/concat_grad/add_9:z:0*
N*
T0*
_output_shapes
:2/
-gradients/concat_grad/strided_slice_9/stack_1?
-gradients/concat_grad/strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-gradients/concat_grad/strided_slice_9/stack_2?
%gradients/concat_grad/strided_slice_9StridedSlice&gradients/concat_grad/Shape_9:output:04gradients/concat_grad/strided_slice_9/stack:output:06gradients/concat_grad/strided_slice_9/stack_1:output:06gradients/concat_grad/strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%gradients/concat_grad/strided_slice_9?
gradients/concat_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB:2 
gradients/concat_grad/Shape_10?
gradients/concat_grad/add_10/yConst*
_output_shapes
: *
dtype0*
value	B :2 
gradients/concat_grad/add_10/y?
gradients/concat_grad/add_10AddV2%gradients_concat_grad_add_concat_axis'gradients/concat_grad/add_10/y:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/add_10?
,gradients/concat_grad/strided_slice_10/stackPack%gradients_concat_grad_add_concat_axis*
N*
T0*
_output_shapes
:2.
,gradients/concat_grad/strided_slice_10/stack?
.gradients/concat_grad/strided_slice_10/stack_1Pack gradients/concat_grad/add_10:z:0*
N*
T0*
_output_shapes
:20
.gradients/concat_grad/strided_slice_10/stack_1?
.gradients/concat_grad/strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.gradients/concat_grad/strided_slice_10/stack_2?
&gradients/concat_grad/strided_slice_10StridedSlice'gradients/concat_grad/Shape_10:output:05gradients/concat_grad/strided_slice_10/stack:output:07gradients/concat_grad/strided_slice_10/stack_1:output:07gradients/concat_grad/strided_slice_10/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2(
&gradients/concat_grad/strided_slice_10?
gradients/concat_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB:2 
gradients/concat_grad/Shape_11?
gradients/concat_grad/add_11/yConst*
_output_shapes
: *
dtype0*
value	B :2 
gradients/concat_grad/add_11/y?
gradients/concat_grad/add_11AddV2%gradients_concat_grad_add_concat_axis'gradients/concat_grad/add_11/y:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/add_11?
,gradients/concat_grad/strided_slice_11/stackPack%gradients_concat_grad_add_concat_axis*
N*
T0*
_output_shapes
:2.
,gradients/concat_grad/strided_slice_11/stack?
.gradients/concat_grad/strided_slice_11/stack_1Pack gradients/concat_grad/add_11:z:0*
N*
T0*
_output_shapes
:20
.gradients/concat_grad/strided_slice_11/stack_1?
.gradients/concat_grad/strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.gradients/concat_grad/strided_slice_11/stack_2?
&gradients/concat_grad/strided_slice_11StridedSlice'gradients/concat_grad/Shape_11:output:05gradients/concat_grad/strided_slice_11/stack:output:07gradients/concat_grad/strided_slice_11/stack_1:output:07gradients/concat_grad/strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2(
&gradients/concat_grad/strided_slice_11?
gradients/concat_grad/packedPack,gradients/concat_grad/strided_slice:output:0.gradients/concat_grad/strided_slice_1:output:0.gradients/concat_grad/strided_slice_2:output:0.gradients/concat_grad/strided_slice_3:output:0.gradients/concat_grad/strided_slice_4:output:0.gradients/concat_grad/strided_slice_5:output:0.gradients/concat_grad/strided_slice_6:output:0.gradients/concat_grad/strided_slice_7:output:0.gradients/concat_grad/strided_slice_8:output:0.gradients/concat_grad/strided_slice_9:output:0/gradients/concat_grad/strided_slice_10:output:0/gradients/concat_grad/strided_slice_11:output:0*
N*
T0*
_output_shapes
:2
gradients/concat_grad/packed?
%gradients/concat_grad/MLCConcatV2GradMLCConcatV2Grad:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0%gradients/concat_grad/packed:output:0%gradients_concat_grad_add_concat_axis$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0&gradients/concat_grad/Shape_2:output:0&gradients/concat_grad/Shape_3:output:0&gradients/concat_grad/Shape_4:output:0&gradients/concat_grad/Shape_5:output:0&gradients/concat_grad/Shape_6:output:0&gradients/concat_grad/Shape_7:output:0&gradients/concat_grad/Shape_8:output:0&gradients/concat_grad/Shape_9:output:0'gradients/concat_grad/Shape_10:output:0'gradients/concat_grad/Shape_11:output:0/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat*
N*
T0*

Tlen0*b
_output_shapesP
N:?:?:?:?:?:?::::::2'
%gradients/concat_grad/MLCConcatV2Grad?
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
gradients/Reshape_1_grad/Shape?
'gradients/Reshape_1_grad/MLCReshapeGradMLCReshapeGrad.gradients/concat_grad/MLCConcatV2Grad:output:0'gradients/Reshape_1_grad/Shape:output:01gradients_reshape_1_grad_mlcreshapegrad_reshape_1*
T0*
_output_shapes

:2)
'gradients/Reshape_1_grad/MLCReshapeGrad?
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
gradients/Reshape_2_grad/Shape?
'gradients/Reshape_2_grad/MLCReshapeGradMLCReshapeGrad.gradients/concat_grad/MLCConcatV2Grad:output:1'gradients/Reshape_2_grad/Shape:output:01gradients_reshape_2_grad_mlcreshapegrad_reshape_2*
T0*
_output_shapes

:2)
'gradients/Reshape_2_grad/MLCReshapeGrad?
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
gradients/Reshape_3_grad/Shape?
'gradients/Reshape_3_grad/MLCReshapeGradMLCReshapeGrad.gradients/concat_grad/MLCConcatV2Grad:output:2'gradients/Reshape_3_grad/Shape:output:01gradients_reshape_3_grad_mlcreshapegrad_reshape_3*
T0*
_output_shapes

:2)
'gradients/Reshape_3_grad/MLCReshapeGrad?
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
gradients/Reshape_4_grad/Shape?
'gradients/Reshape_4_grad/MLCReshapeGradMLCReshapeGrad.gradients/concat_grad/MLCConcatV2Grad:output:3'gradients/Reshape_4_grad/Shape:output:01gradients_reshape_4_grad_mlcreshapegrad_reshape_4*
T0*
_output_shapes

:2)
'gradients/Reshape_4_grad/MLCReshapeGrad?
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
gradients/Reshape_5_grad/Shape?
'gradients/Reshape_5_grad/MLCReshapeGradMLCReshapeGrad.gradients/concat_grad/MLCConcatV2Grad:output:4'gradients/Reshape_5_grad/Shape:output:01gradients_reshape_5_grad_mlcreshapegrad_reshape_5*
T0*
_output_shapes

:2)
'gradients/Reshape_5_grad/MLCReshapeGrad?
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
gradients/Reshape_6_grad/Shape?
'gradients/Reshape_6_grad/MLCReshapeGradMLCReshapeGrad.gradients/concat_grad/MLCConcatV2Grad:output:5'gradients/Reshape_6_grad/Shape:output:01gradients_reshape_6_grad_mlcreshapegrad_reshape_6*
T0*
_output_shapes

:2)
'gradients/Reshape_6_grad/MLCReshapeGrad?
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2 
gradients/Reshape_7_grad/Shape?
'gradients/Reshape_7_grad/MLCReshapeGradMLCReshapeGrad.gradients/concat_grad/MLCConcatV2Grad:output:6'gradients/Reshape_7_grad/Shape:output:01gradients_reshape_7_grad_mlcreshapegrad_reshape_7*
T0*
_output_shapes
:2)
'gradients/Reshape_7_grad/MLCReshapeGrad?
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2 
gradients/Reshape_8_grad/Shape?
'gradients/Reshape_8_grad/MLCReshapeGradMLCReshapeGrad.gradients/concat_grad/MLCConcatV2Grad:output:7'gradients/Reshape_8_grad/Shape:output:01gradients_reshape_8_grad_mlcreshapegrad_reshape_8*
T0*
_output_shapes
:2)
'gradients/Reshape_8_grad/MLCReshapeGrad?
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2 
gradients/Reshape_9_grad/Shape?
'gradients/Reshape_9_grad/MLCReshapeGradMLCReshapeGrad.gradients/concat_grad/MLCConcatV2Grad:output:8'gradients/Reshape_9_grad/Shape:output:01gradients_reshape_9_grad_mlcreshapegrad_reshape_9*
T0*
_output_shapes
:2)
'gradients/Reshape_9_grad/MLCReshapeGrad?
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2!
gradients/Reshape_10_grad/Shape?
(gradients/Reshape_10_grad/MLCReshapeGradMLCReshapeGrad.gradients/concat_grad/MLCConcatV2Grad:output:9(gradients/Reshape_10_grad/Shape:output:03gradients_reshape_10_grad_mlcreshapegrad_reshape_10*
T0*
_output_shapes
:2*
(gradients/Reshape_10_grad/MLCReshapeGrad?
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2!
gradients/Reshape_11_grad/Shape?
(gradients/Reshape_11_grad/MLCReshapeGradMLCReshapeGrad/gradients/concat_grad/MLCConcatV2Grad:output:10(gradients/Reshape_11_grad/Shape:output:03gradients_reshape_11_grad_mlcreshapegrad_reshape_11*
T0*
_output_shapes
:2*
(gradients/Reshape_11_grad/MLCReshapeGrad?
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2!
gradients/Reshape_12_grad/Shape?
(gradients/Reshape_12_grad/MLCReshapeGradMLCReshapeGrad/gradients/concat_grad/MLCConcatV2Grad:output:11(gradients/Reshape_12_grad/Shape:output:03gradients_reshape_12_grad_mlcreshapegrad_reshape_12*
T0*
_output_shapes
:2*
(gradients/Reshape_12_grad/MLCReshapeGrad?
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:2.
,gradients/transpose_1_grad/InvertPermutation?
+gradients/transpose_1_grad/MLCTransposeGradMLCTransposeGrad0gradients/Reshape_1_grad/MLCReshapeGrad:output:00gradients/transpose_1_grad/InvertPermutation:y:07gradients_transpose_1_grad_mlctransposegrad_transpose_1*
T0*
_output_shapes

:2-
+gradients/transpose_1_grad/MLCTransposeGrad?
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:2.
,gradients/transpose_2_grad/InvertPermutation?
+gradients/transpose_2_grad/MLCTransposeGradMLCTransposeGrad0gradients/Reshape_2_grad/MLCReshapeGrad:output:00gradients/transpose_2_grad/InvertPermutation:y:07gradients_transpose_2_grad_mlctransposegrad_transpose_2*
T0*
_output_shapes

:2-
+gradients/transpose_2_grad/MLCTransposeGrad?
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:2.
,gradients/transpose_3_grad/InvertPermutation?
+gradients/transpose_3_grad/MLCTransposeGradMLCTransposeGrad0gradients/Reshape_3_grad/MLCReshapeGrad:output:00gradients/transpose_3_grad/InvertPermutation:y:07gradients_transpose_3_grad_mlctransposegrad_transpose_3*
T0*
_output_shapes

:2-
+gradients/transpose_3_grad/MLCTransposeGrad?
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:2.
,gradients/transpose_4_grad/InvertPermutation?
+gradients/transpose_4_grad/MLCTransposeGradMLCTransposeGrad0gradients/Reshape_4_grad/MLCReshapeGrad:output:00gradients/transpose_4_grad/InvertPermutation:y:07gradients_transpose_4_grad_mlctransposegrad_transpose_4*
T0*
_output_shapes

:2-
+gradients/transpose_4_grad/MLCTransposeGrad?
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:2.
,gradients/transpose_5_grad/InvertPermutation?
+gradients/transpose_5_grad/MLCTransposeGradMLCTransposeGrad0gradients/Reshape_5_grad/MLCReshapeGrad:output:00gradients/transpose_5_grad/InvertPermutation:y:07gradients_transpose_5_grad_mlctransposegrad_transpose_5*
T0*
_output_shapes

:2-
+gradients/transpose_5_grad/MLCTransposeGrad?
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:2.
,gradients/transpose_6_grad/InvertPermutation?
+gradients/transpose_6_grad/MLCTransposeGradMLCTransposeGrad0gradients/Reshape_6_grad/MLCReshapeGrad:output:00gradients/transpose_6_grad/InvertPermutation:y:07gradients_transpose_6_grad_mlctransposegrad_transpose_6*
T0*
_output_shapes

:2-
+gradients/transpose_6_grad/MLCTransposeGrad?
gradients/split_2_grad/concatConcatV20gradients/Reshape_7_grad/MLCReshapeGrad:output:00gradients/Reshape_8_grad/MLCReshapeGrad:output:00gradients/Reshape_9_grad/MLCReshapeGrad:output:01gradients/Reshape_10_grad/MLCReshapeGrad:output:01gradients/Reshape_11_grad/MLCReshapeGrad:output:01gradients/Reshape_12_grad/MLCReshapeGrad:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes	
:?2
gradients/split_2_grad/concat?
!gradients/split_grad/MLCSplitGradMLCSplitGrad/gradients/transpose_1_grad/MLCTransposeGrad:y:0/gradients/transpose_2_grad/MLCTransposeGrad:y:0/gradients/transpose_3_grad/MLCTransposeGrad:y:01gradients_split_grad_mlcsplitgrad_split_split_dim'gradients_split_grad_mlcsplitgrad_split*
N*
T0*
_output_shapes

:Z2#
!gradients/split_grad/MLCSplitGrad?
#gradients/split_1_grad/MLCSplitGradMLCSplitGrad/gradients/transpose_4_grad/MLCTransposeGrad:y:0/gradients/transpose_5_grad/MLCTransposeGrad:y:0/gradients/transpose_6_grad/MLCTransposeGrad:y:05gradients_split_1_grad_mlcsplitgrad_split_1_split_dim+gradients_split_1_grad_mlcsplitgrad_split_1*
N*
T0*
_output_shapes

:Z2%
#gradients/split_1_grad/MLCSplitGrad?
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"   Z   2
gradients/Reshape_grad/Shape?
%gradients/Reshape_grad/MLCReshapeGradMLCReshapeGrad&gradients/split_2_grad/concat:output:0%gradients/Reshape_grad/Shape:output:0-gradients_reshape_grad_mlcreshapegrad_reshape*
T0*
_output_shapes

:Z2'
%gradients/Reshape_grad/MLCReshapeGrad?
IdentityIdentity-gradients/transpose_grad/MLCTransposeGrad:y:0*
T0*,
_output_shapes
:??????????2

Identity?

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0*
T0*'
_output_shapes
:?????????2

Identity_1y

Identity_2Identity*gradients/split_grad/MLCSplitGrad:output:0*
T0*
_output_shapes

:Z2

Identity_2{

Identity_3Identity,gradients/split_1_grad/MLCSplitGrad:output:0*
T0*
_output_shapes

:Z2

Identity_3}

Identity_4Identity.gradients/Reshape_grad/MLCReshapeGrad:output:0*
T0*
_output_shapes

:Z2

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*?
_input_shapes?
?:?????????:??????????:?????????: :??????????:?????????::??????????:?????????: ::??????????:?????????: :?::?????????: :?:?:?:?:?:?::::::::::::::::::: : :: ::?*<
api_implements*(gru_e9d87f5a-6bce-4ea5-b775-fb635bd39d20*
api_preferred_deviceGPU*@
forward_function_name'%__forward_gpu_gru_with_fallback_19705*
go_backwards( *

time_major( :- )
'
_output_shapes
:?????????:2.
,
_output_shapes
:??????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :2.
,
_output_shapes
:??????????:-)
'
_output_shapes
:?????????: 

_output_shapes
::2.
,
_output_shapes
:??????????:1-
+
_output_shapes
:?????????:	

_output_shapes
: :


_output_shapes
::2.
,
_output_shapes
:??????????:1-
+
_output_shapes
:?????????:

_output_shapes
: :!

_output_shapes	
:?: 

_output_shapes
::-)
'
_output_shapes
:?????????:

_output_shapes
: :!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::$ 

_output_shapes

::  

_output_shapes
::$! 

_output_shapes

:: "

_output_shapes
::$# 

_output_shapes

:: $

_output_shapes
::$% 

_output_shapes

:: &

_output_shapes
::$' 

_output_shapes

:: (

_output_shapes
::$) 

_output_shapes

::*

_output_shapes
: :+

_output_shapes
: :$, 

_output_shapes

::-

_output_shapes
: :$. 

_output_shapes

::!/

_output_shapes	
:?
?
?
G__inference_sequential_1_layer_call_and_return_conditional_losses_17259

inputs
	gru_17246
	gru_17248
	gru_17250
dense_2_17253
dense_2_17255
identity??dense_2/StatefulPartitionedCall?gru/StatefulPartitionedCall?
gru/StatefulPartitionedCallStatefulPartitionedCallinputs	gru_17246	gru_17248	gru_17250*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *G
fBR@
>__inference_gru_layer_call_and_return_conditional_losses_171352
gru/StatefulPartitionedCall?
dense_2/StatefulPartitionedCallStatefulPartitionedCall$gru/StatefulPartitionedCall:output:0dense_2_17253dense_2_17255*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_171762!
dense_2/StatefulPartitionedCall?
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0 ^dense_2/StatefulPartitionedCall^gru/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????:::::2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2:
gru/StatefulPartitionedCallgru/StatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?2
?
while_body_15794
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_matmul_kernel_0
while_biasadd_unstack_0%
!while_matmul_1_recurrent_kernel_0
while_biasadd_1_unstack_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_matmul_kernel
while_biasadd_unstack#
while_matmul_1_recurrent_kernel
while_biasadd_1_unstack?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
while/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while_matmul_kernel_0*
T0*'
_output_shapes
:?????????Z2
while/MatMul?
while/BiasAddBiasAddwhile/MatMul:product:0while_biasadd_unstack_0*
T0*'
_output_shapes
:?????????Z2
while/BiasAdd\
while/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/Constp
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
while/split/split_dim?
while/splitSplitwhile/split/split_dim:output:0while/BiasAdd:output:0*
T0*M
_output_shapes;
9:?????????:?????????:?????????*
	num_split2
while/split?
while/MatMul_1MatMulwhile_placeholder_2!while_matmul_1_recurrent_kernel_0*
T0*'
_output_shapes
:?????????Z2
while/MatMul_1?
while/BiasAdd_1BiasAddwhile/MatMul_1:product:0while_biasadd_1_unstack_0*
T0*'
_output_shapes
:?????????Z2
while/BiasAdd_1`
while/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
while/Const_1t
while/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
while/split_1/split_dim?
while/split_1Split while/split_1/split_dim:output:0while/BiasAdd_1:output:0*
T0*M
_output_shapes;
9:?????????:?????????:?????????*
	num_split2
while/split_1
	while/addAddV2while/split:output:0while/split_1:output:0*
T0*'
_output_shapes
:?????????2
	while/addj
while/SigmoidSigmoidwhile/add:z:0*
T0*'
_output_shapes
:?????????2
while/Sigmoid?
while/add_1AddV2while/split:output:1while/split_1:output:1*
T0*'
_output_shapes
:?????????2
while/add_1p
while/Sigmoid_1Sigmoidwhile/add_1:z:0*
T0*'
_output_shapes
:?????????2
while/Sigmoid_1|
	while/mulMulwhile/Sigmoid_1:y:0while/split_1:output:2*
T0*'
_output_shapes
:?????????2
	while/mulz
while/add_2AddV2while/split:output:2while/mul:z:0*
T0*'
_output_shapes
:?????????2
while/add_2c

while/TanhTanhwhile/add_2:z:0*
T0*'
_output_shapes
:?????????2

while/Tanh{
while/mul_1Mulwhile/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:?????????2
while/mul_1_
while/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
while/sub/xx
	while/subSubwhile/sub/x:output:0while/Sigmoid:y:0*
T0*'
_output_shapes
:?????????2
	while/subr
while/mul_2Mulwhile/sub:z:0while/Tanh:y:0*
T0*'
_output_shapes
:?????????2
while/mul_2w
while/add_3AddV2while/mul_1:z:0while/mul_2:z:0*
T0*'
_output_shapes
:?????????2
while/add_3?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/add_3:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem`
while/add_4/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_4/yo
while/add_4AddV2while_placeholderwhile/add_4/y:output:0*
T0*
_output_shapes
: 2
while/add_4`
while/add_5/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_5/yv
while/add_5AddV2while_while_loop_counterwhile/add_5/y:output:0*
T0*
_output_shapes
: 2
while/add_5^
while/IdentityIdentitywhile/add_5:z:0*
T0*
_output_shapes
: 2
while/Identityq
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: 2
while/Identity_1b
while/Identity_2Identitywhile/add_4:z:0*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3s
while/Identity_4Identitywhile/add_3:z:0*
T0*'
_output_shapes
:?????????2
while/Identity_4"4
while_biasadd_1_unstackwhile_biasadd_1_unstack_0"0
while_biasadd_unstackwhile_biasadd_unstack_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"D
while_matmul_1_recurrent_kernel!while_matmul_1_recurrent_kernel_0",
while_matmul_kernelwhile_matmul_kernel_0",
while_strided_slicewhile_strided_slice_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*R
_input_shapesA
?: : : : :?????????: : :Z:Z:Z:Z: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:Z: 

_output_shapes
:Z:$	 

_output_shapes

:Z: 


_output_shapes
:Z
?	
?
while_cond_16743
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice3
/while_while_cond_16743___redundant_placeholder03
/while_while_cond_16743___redundant_placeholder13
/while_while_cond_16743___redundant_placeholder23
/while_while_cond_16743___redundant_placeholder33
/while_while_cond_16743___redundant_placeholder4
while_identity
n

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*D
_input_shapes3
1: : : : :?????????: :::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
::

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
?
?
#__inference_signature_wrapper_17297
	gru_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCall	gru_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8? *)
f$R"
 __inference__wrapped_model_142942
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????:::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
,
_output_shapes
:??????????
#
_user_specified_name	gru_input
?
?
,__inference_sequential_1_layer_call_fn_18279

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_172592
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????:::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
,__inference_sequential_1_layer_call_fn_17241
	gru_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCall	gru_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_172282
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????:::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
,
_output_shapes
:??????????
#
_user_specified_name	gru_input
??
?
8__inference___backward_gpu_gru_with_fallback_16446_16664
placeholder
placeholder_1
placeholder_2
placeholder_3/
+gradients_strided_slice_grad_shape_cudnnrnnB
>gradients_strided_slice_grad_mlcstridedslicegrad_strided_sliceA
=gradients_transpose_7_grad_invertpermutation_transpose_7_perm;
7gradients_transpose_7_grad_mlctransposegrad_transpose_7)
%gradients_squeeze_grad_shape_cudnnrnn!
gradients_zeros_like_cudnnrnn#
gradients_zeros_like_1_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims=
9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c3
/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h)
%gradients_concat_grad_add_concat_axis5
1gradients_reshape_1_grad_mlcreshapegrad_reshape_15
1gradients_reshape_2_grad_mlcreshapegrad_reshape_25
1gradients_reshape_3_grad_mlcreshapegrad_reshape_35
1gradients_reshape_4_grad_mlcreshapegrad_reshape_45
1gradients_reshape_5_grad_mlcreshapegrad_reshape_55
1gradients_reshape_6_grad_mlcreshapegrad_reshape_65
1gradients_reshape_7_grad_mlcreshapegrad_reshape_75
1gradients_reshape_8_grad_mlcreshapegrad_reshape_85
1gradients_reshape_9_grad_mlcreshapegrad_reshape_97
3gradients_reshape_10_grad_mlcreshapegrad_reshape_107
3gradients_reshape_11_grad_mlcreshapegrad_reshape_117
3gradients_reshape_12_grad_mlcreshapegrad_reshape_12A
=gradients_transpose_1_grad_invertpermutation_transpose_1_perm;
7gradients_transpose_1_grad_mlctransposegrad_transpose_1A
=gradients_transpose_2_grad_invertpermutation_transpose_2_perm;
7gradients_transpose_2_grad_mlctransposegrad_transpose_2A
=gradients_transpose_3_grad_invertpermutation_transpose_3_perm;
7gradients_transpose_3_grad_mlctransposegrad_transpose_3A
=gradients_transpose_4_grad_invertpermutation_transpose_4_perm;
7gradients_transpose_4_grad_mlctransposegrad_transpose_4A
=gradients_transpose_5_grad_invertpermutation_transpose_5_perm;
7gradients_transpose_5_grad_mlctransposegrad_transpose_5A
=gradients_transpose_6_grad_invertpermutation_transpose_6_perm;
7gradients_transpose_6_grad_mlctransposegrad_transpose_63
/gradients_split_2_grad_concat_split_2_split_dim5
1gradients_split_grad_mlcsplitgrad_split_split_dim+
'gradients_split_grad_mlcsplitgrad_split9
5gradients_split_1_grad_mlcsplitgrad_split_1_split_dim/
+gradients_split_1_grad_mlcsplitgrad_split_11
-gradients_reshape_grad_mlcreshapegrad_reshape
identity

identity_1

identity_2

identity_3

identity_4?u
gradients/grad_ys_0Identityplaceholder*
T0*'
_output_shapes
:?????????2
gradients/grad_ys_0|
gradients/grad_ys_1Identityplaceholder_1*
T0*,
_output_shapes
:??????????2
gradients/grad_ys_1w
gradients/grad_ys_2Identityplaceholder_2*
T0*'
_output_shapes
:?????????2
gradients/grad_ys_2f
gradients/grad_ys_3Identityplaceholder_3*
T0*
_output_shapes
: 2
gradients/grad_ys_3?
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:2$
"gradients/strided_slice_grad/Shape?
6gradients/strided_slice_grad/MLCStridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*
valueB:
?????????28
6gradients/strided_slice_grad/MLCStridedSliceGrad/begin?
4gradients/strided_slice_grad/MLCStridedSliceGrad/endConst*
_output_shapes
:*
dtype0*
valueB: 26
4gradients/strided_slice_grad/MLCStridedSliceGrad/end?
8gradients/strided_slice_grad/MLCStridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*
valueB:2:
8gradients/strided_slice_grad/MLCStridedSliceGrad/strides?
0gradients/strided_slice_grad/MLCStridedSliceGradMLCStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0?gradients/strided_slice_grad/MLCStridedSliceGrad/begin:output:0=gradients/strided_slice_grad/MLCStridedSliceGrad/end:output:0Agradients/strided_slice_grad/MLCStridedSliceGrad/strides:output:0gradients/grad_ys_0:output:0>gradients_strided_slice_grad_mlcstridedslicegrad_strided_slice*
Index0*
T0*,
_output_shapes
:??????????*
shrink_axis_mask22
0gradients/strided_slice_grad/MLCStridedSliceGrad?
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:2.
,gradients/transpose_7_grad/InvertPermutation?
+gradients/transpose_7_grad/MLCTransposeGradMLCTransposeGradgradients/grad_ys_1:output:00gradients/transpose_7_grad/InvertPermutation:y:07gradients_transpose_7_grad_mlctransposegrad_transpose_7*
T0*,
_output_shapes
:??????????2-
+gradients/transpose_7_grad/MLCTransposeGrad?
gradients/Squeeze_grad/ShapeShape%gradients_squeeze_grad_shape_cudnnrnn*
T0*
_output_shapes
:2
gradients/Squeeze_grad/Shape?
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*+
_output_shapes
:?????????2 
gradients/Squeeze_grad/Reshape?
gradients/AddNAddN9gradients/strided_slice_grad/MLCStridedSliceGrad:output:0/gradients/transpose_7_grad/MLCTransposeGrad:y:0*
N*
T0*C
_class9
75loc:@gradients/strided_slice_grad/MLCStridedSliceGrad*,
_output_shapes
:??????????2
gradients/AddNy
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
: 2
gradients/zeros_like?
gradients/zeros_like_1	ZerosLikegradients_zeros_like_1_cudnnrnn*
T0*
_output_shapes
:2
gradients/zeros_like_1?
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat+gradients_strided_slice_grad_shape_cudnnrnn%gradients_squeeze_grad_shape_cudnnrnngradients_zeros_like_cudnnrnngradients/AddN:sum:0'gradients/Squeeze_grad/Reshape:output:0gradients/zeros_like:y:0gradients_zeros_like_1_cudnnrnn*
T0*L
_output_shapes:
8:??????????:?????????: :?*
rnn_modegru2*
(gradients/CudnnRNN_grad/CudnnRNNBackprop?
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:2,
*gradients/transpose_grad/InvertPermutation?
)gradients/transpose_grad/MLCTransposeGradMLCTransposeGrad9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:02gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose*
T0*,
_output_shapes
:??????????2+
)gradients/transpose_grad/MLCTransposeGrad?
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
:2!
gradients/ExpandDims_grad/Shape?
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*'
_output_shapes
:?????????2#
!gradients/ExpandDims_grad/Reshape?
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:?2
gradients/concat_grad/Shape|
gradients/concat_grad/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_grad/add/y?
gradients/concat_grad/addAddV2%gradients_concat_grad_add_concat_axis$gradients/concat_grad/add/y:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/add?
)gradients/concat_grad/strided_slice/stackPack%gradients_concat_grad_add_concat_axis*
N*
T0*
_output_shapes
:2+
)gradients/concat_grad/strided_slice/stack?
+gradients/concat_grad/strided_slice/stack_1Packgradients/concat_grad/add:z:0*
N*
T0*
_output_shapes
:2-
+gradients/concat_grad/strided_slice/stack_1?
+gradients/concat_grad/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+gradients/concat_grad/strided_slice/stack_2?
#gradients/concat_grad/strided_sliceStridedSlice$gradients/concat_grad/Shape:output:02gradients/concat_grad/strided_slice/stack:output:04gradients/concat_grad/strided_slice/stack_1:output:04gradients/concat_grad/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#gradients/concat_grad/strided_slice?
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:?2
gradients/concat_grad/Shape_1?
gradients/concat_grad/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_grad/add_1/y?
gradients/concat_grad/add_1AddV2%gradients_concat_grad_add_concat_axis&gradients/concat_grad/add_1/y:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/add_1?
+gradients/concat_grad/strided_slice_1/stackPack%gradients_concat_grad_add_concat_axis*
N*
T0*
_output_shapes
:2-
+gradients/concat_grad/strided_slice_1/stack?
-gradients/concat_grad/strided_slice_1/stack_1Packgradients/concat_grad/add_1:z:0*
N*
T0*
_output_shapes
:2/
-gradients/concat_grad/strided_slice_1/stack_1?
-gradients/concat_grad/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-gradients/concat_grad/strided_slice_1/stack_2?
%gradients/concat_grad/strided_slice_1StridedSlice&gradients/concat_grad/Shape_1:output:04gradients/concat_grad/strided_slice_1/stack:output:06gradients/concat_grad/strided_slice_1/stack_1:output:06gradients/concat_grad/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%gradients/concat_grad/strided_slice_1?
gradients/concat_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:?2
gradients/concat_grad/Shape_2?
gradients/concat_grad/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_grad/add_2/y?
gradients/concat_grad/add_2AddV2%gradients_concat_grad_add_concat_axis&gradients/concat_grad/add_2/y:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/add_2?
+gradients/concat_grad/strided_slice_2/stackPack%gradients_concat_grad_add_concat_axis*
N*
T0*
_output_shapes
:2-
+gradients/concat_grad/strided_slice_2/stack?
-gradients/concat_grad/strided_slice_2/stack_1Packgradients/concat_grad/add_2:z:0*
N*
T0*
_output_shapes
:2/
-gradients/concat_grad/strided_slice_2/stack_1?
-gradients/concat_grad/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-gradients/concat_grad/strided_slice_2/stack_2?
%gradients/concat_grad/strided_slice_2StridedSlice&gradients/concat_grad/Shape_2:output:04gradients/concat_grad/strided_slice_2/stack:output:06gradients/concat_grad/strided_slice_2/stack_1:output:06gradients/concat_grad/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%gradients/concat_grad/strided_slice_2?
gradients/concat_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:?2
gradients/concat_grad/Shape_3?
gradients/concat_grad/add_3/yConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_grad/add_3/y?
gradients/concat_grad/add_3AddV2%gradients_concat_grad_add_concat_axis&gradients/concat_grad/add_3/y:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/add_3?
+gradients/concat_grad/strided_slice_3/stackPack%gradients_concat_grad_add_concat_axis*
N*
T0*
_output_shapes
:2-
+gradients/concat_grad/strided_slice_3/stack?
-gradients/concat_grad/strided_slice_3/stack_1Packgradients/concat_grad/add_3:z:0*
N*
T0*
_output_shapes
:2/
-gradients/concat_grad/strided_slice_3/stack_1?
-gradients/concat_grad/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-gradients/concat_grad/strided_slice_3/stack_2?
%gradients/concat_grad/strided_slice_3StridedSlice&gradients/concat_grad/Shape_3:output:04gradients/concat_grad/strided_slice_3/stack:output:06gradients/concat_grad/strided_slice_3/stack_1:output:06gradients/concat_grad/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%gradients/concat_grad/strided_slice_3?
gradients/concat_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:?2
gradients/concat_grad/Shape_4?
gradients/concat_grad/add_4/yConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_grad/add_4/y?
gradients/concat_grad/add_4AddV2%gradients_concat_grad_add_concat_axis&gradients/concat_grad/add_4/y:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/add_4?
+gradients/concat_grad/strided_slice_4/stackPack%gradients_concat_grad_add_concat_axis*
N*
T0*
_output_shapes
:2-
+gradients/concat_grad/strided_slice_4/stack?
-gradients/concat_grad/strided_slice_4/stack_1Packgradients/concat_grad/add_4:z:0*
N*
T0*
_output_shapes
:2/
-gradients/concat_grad/strided_slice_4/stack_1?
-gradients/concat_grad/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-gradients/concat_grad/strided_slice_4/stack_2?
%gradients/concat_grad/strided_slice_4StridedSlice&gradients/concat_grad/Shape_4:output:04gradients/concat_grad/strided_slice_4/stack:output:06gradients/concat_grad/strided_slice_4/stack_1:output:06gradients/concat_grad/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%gradients/concat_grad/strided_slice_4?
gradients/concat_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:?2
gradients/concat_grad/Shape_5?
gradients/concat_grad/add_5/yConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_grad/add_5/y?
gradients/concat_grad/add_5AddV2%gradients_concat_grad_add_concat_axis&gradients/concat_grad/add_5/y:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/add_5?
+gradients/concat_grad/strided_slice_5/stackPack%gradients_concat_grad_add_concat_axis*
N*
T0*
_output_shapes
:2-
+gradients/concat_grad/strided_slice_5/stack?
-gradients/concat_grad/strided_slice_5/stack_1Packgradients/concat_grad/add_5:z:0*
N*
T0*
_output_shapes
:2/
-gradients/concat_grad/strided_slice_5/stack_1?
-gradients/concat_grad/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-gradients/concat_grad/strided_slice_5/stack_2?
%gradients/concat_grad/strided_slice_5StridedSlice&gradients/concat_grad/Shape_5:output:04gradients/concat_grad/strided_slice_5/stack:output:06gradients/concat_grad/strided_slice_5/stack_1:output:06gradients/concat_grad/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%gradients/concat_grad/strided_slice_5?
gradients/concat_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape_6?
gradients/concat_grad/add_6/yConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_grad/add_6/y?
gradients/concat_grad/add_6AddV2%gradients_concat_grad_add_concat_axis&gradients/concat_grad/add_6/y:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/add_6?
+gradients/concat_grad/strided_slice_6/stackPack%gradients_concat_grad_add_concat_axis*
N*
T0*
_output_shapes
:2-
+gradients/concat_grad/strided_slice_6/stack?
-gradients/concat_grad/strided_slice_6/stack_1Packgradients/concat_grad/add_6:z:0*
N*
T0*
_output_shapes
:2/
-gradients/concat_grad/strided_slice_6/stack_1?
-gradients/concat_grad/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-gradients/concat_grad/strided_slice_6/stack_2?
%gradients/concat_grad/strided_slice_6StridedSlice&gradients/concat_grad/Shape_6:output:04gradients/concat_grad/strided_slice_6/stack:output:06gradients/concat_grad/strided_slice_6/stack_1:output:06gradients/concat_grad/strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%gradients/concat_grad/strided_slice_6?
gradients/concat_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape_7?
gradients/concat_grad/add_7/yConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_grad/add_7/y?
gradients/concat_grad/add_7AddV2%gradients_concat_grad_add_concat_axis&gradients/concat_grad/add_7/y:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/add_7?
+gradients/concat_grad/strided_slice_7/stackPack%gradients_concat_grad_add_concat_axis*
N*
T0*
_output_shapes
:2-
+gradients/concat_grad/strided_slice_7/stack?
-gradients/concat_grad/strided_slice_7/stack_1Packgradients/concat_grad/add_7:z:0*
N*
T0*
_output_shapes
:2/
-gradients/concat_grad/strided_slice_7/stack_1?
-gradients/concat_grad/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-gradients/concat_grad/strided_slice_7/stack_2?
%gradients/concat_grad/strided_slice_7StridedSlice&gradients/concat_grad/Shape_7:output:04gradients/concat_grad/strided_slice_7/stack:output:06gradients/concat_grad/strided_slice_7/stack_1:output:06gradients/concat_grad/strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%gradients/concat_grad/strided_slice_7?
gradients/concat_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape_8?
gradients/concat_grad/add_8/yConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_grad/add_8/y?
gradients/concat_grad/add_8AddV2%gradients_concat_grad_add_concat_axis&gradients/concat_grad/add_8/y:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/add_8?
+gradients/concat_grad/strided_slice_8/stackPack%gradients_concat_grad_add_concat_axis*
N*
T0*
_output_shapes
:2-
+gradients/concat_grad/strided_slice_8/stack?
-gradients/concat_grad/strided_slice_8/stack_1Packgradients/concat_grad/add_8:z:0*
N*
T0*
_output_shapes
:2/
-gradients/concat_grad/strided_slice_8/stack_1?
-gradients/concat_grad/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-gradients/concat_grad/strided_slice_8/stack_2?
%gradients/concat_grad/strided_slice_8StridedSlice&gradients/concat_grad/Shape_8:output:04gradients/concat_grad/strided_slice_8/stack:output:06gradients/concat_grad/strided_slice_8/stack_1:output:06gradients/concat_grad/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%gradients/concat_grad/strided_slice_8?
gradients/concat_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape_9?
gradients/concat_grad/add_9/yConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_grad/add_9/y?
gradients/concat_grad/add_9AddV2%gradients_concat_grad_add_concat_axis&gradients/concat_grad/add_9/y:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/add_9?
+gradients/concat_grad/strided_slice_9/stackPack%gradients_concat_grad_add_concat_axis*
N*
T0*
_output_shapes
:2-
+gradients/concat_grad/strided_slice_9/stack?
-gradients/concat_grad/strided_slice_9/stack_1Packgradients/concat_grad/add_9:z:0*
N*
T0*
_output_shapes
:2/
-gradients/concat_grad/strided_slice_9/stack_1?
-gradients/concat_grad/strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-gradients/concat_grad/strided_slice_9/stack_2?
%gradients/concat_grad/strided_slice_9StridedSlice&gradients/concat_grad/Shape_9:output:04gradients/concat_grad/strided_slice_9/stack:output:06gradients/concat_grad/strided_slice_9/stack_1:output:06gradients/concat_grad/strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%gradients/concat_grad/strided_slice_9?
gradients/concat_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB:2 
gradients/concat_grad/Shape_10?
gradients/concat_grad/add_10/yConst*
_output_shapes
: *
dtype0*
value	B :2 
gradients/concat_grad/add_10/y?
gradients/concat_grad/add_10AddV2%gradients_concat_grad_add_concat_axis'gradients/concat_grad/add_10/y:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/add_10?
,gradients/concat_grad/strided_slice_10/stackPack%gradients_concat_grad_add_concat_axis*
N*
T0*
_output_shapes
:2.
,gradients/concat_grad/strided_slice_10/stack?
.gradients/concat_grad/strided_slice_10/stack_1Pack gradients/concat_grad/add_10:z:0*
N*
T0*
_output_shapes
:20
.gradients/concat_grad/strided_slice_10/stack_1?
.gradients/concat_grad/strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.gradients/concat_grad/strided_slice_10/stack_2?
&gradients/concat_grad/strided_slice_10StridedSlice'gradients/concat_grad/Shape_10:output:05gradients/concat_grad/strided_slice_10/stack:output:07gradients/concat_grad/strided_slice_10/stack_1:output:07gradients/concat_grad/strided_slice_10/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2(
&gradients/concat_grad/strided_slice_10?
gradients/concat_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB:2 
gradients/concat_grad/Shape_11?
gradients/concat_grad/add_11/yConst*
_output_shapes
: *
dtype0*
value	B :2 
gradients/concat_grad/add_11/y?
gradients/concat_grad/add_11AddV2%gradients_concat_grad_add_concat_axis'gradients/concat_grad/add_11/y:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/add_11?
,gradients/concat_grad/strided_slice_11/stackPack%gradients_concat_grad_add_concat_axis*
N*
T0*
_output_shapes
:2.
,gradients/concat_grad/strided_slice_11/stack?
.gradients/concat_grad/strided_slice_11/stack_1Pack gradients/concat_grad/add_11:z:0*
N*
T0*
_output_shapes
:20
.gradients/concat_grad/strided_slice_11/stack_1?
.gradients/concat_grad/strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.gradients/concat_grad/strided_slice_11/stack_2?
&gradients/concat_grad/strided_slice_11StridedSlice'gradients/concat_grad/Shape_11:output:05gradients/concat_grad/strided_slice_11/stack:output:07gradients/concat_grad/strided_slice_11/stack_1:output:07gradients/concat_grad/strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2(
&gradients/concat_grad/strided_slice_11?
gradients/concat_grad/packedPack,gradients/concat_grad/strided_slice:output:0.gradients/concat_grad/strided_slice_1:output:0.gradients/concat_grad/strided_slice_2:output:0.gradients/concat_grad/strided_slice_3:output:0.gradients/concat_grad/strided_slice_4:output:0.gradients/concat_grad/strided_slice_5:output:0.gradients/concat_grad/strided_slice_6:output:0.gradients/concat_grad/strided_slice_7:output:0.gradients/concat_grad/strided_slice_8:output:0.gradients/concat_grad/strided_slice_9:output:0/gradients/concat_grad/strided_slice_10:output:0/gradients/concat_grad/strided_slice_11:output:0*
N*
T0*
_output_shapes
:2
gradients/concat_grad/packed?
%gradients/concat_grad/MLCConcatV2GradMLCConcatV2Grad:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0%gradients/concat_grad/packed:output:0%gradients_concat_grad_add_concat_axis$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0&gradients/concat_grad/Shape_2:output:0&gradients/concat_grad/Shape_3:output:0&gradients/concat_grad/Shape_4:output:0&gradients/concat_grad/Shape_5:output:0&gradients/concat_grad/Shape_6:output:0&gradients/concat_grad/Shape_7:output:0&gradients/concat_grad/Shape_8:output:0&gradients/concat_grad/Shape_9:output:0'gradients/concat_grad/Shape_10:output:0'gradients/concat_grad/Shape_11:output:0/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat*
N*
T0*

Tlen0*b
_output_shapesP
N:?:?:?:?:?:?::::::2'
%gradients/concat_grad/MLCConcatV2Grad?
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
gradients/Reshape_1_grad/Shape?
'gradients/Reshape_1_grad/MLCReshapeGradMLCReshapeGrad.gradients/concat_grad/MLCConcatV2Grad:output:0'gradients/Reshape_1_grad/Shape:output:01gradients_reshape_1_grad_mlcreshapegrad_reshape_1*
T0*
_output_shapes

:2)
'gradients/Reshape_1_grad/MLCReshapeGrad?
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
gradients/Reshape_2_grad/Shape?
'gradients/Reshape_2_grad/MLCReshapeGradMLCReshapeGrad.gradients/concat_grad/MLCConcatV2Grad:output:1'gradients/Reshape_2_grad/Shape:output:01gradients_reshape_2_grad_mlcreshapegrad_reshape_2*
T0*
_output_shapes

:2)
'gradients/Reshape_2_grad/MLCReshapeGrad?
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
gradients/Reshape_3_grad/Shape?
'gradients/Reshape_3_grad/MLCReshapeGradMLCReshapeGrad.gradients/concat_grad/MLCConcatV2Grad:output:2'gradients/Reshape_3_grad/Shape:output:01gradients_reshape_3_grad_mlcreshapegrad_reshape_3*
T0*
_output_shapes

:2)
'gradients/Reshape_3_grad/MLCReshapeGrad?
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
gradients/Reshape_4_grad/Shape?
'gradients/Reshape_4_grad/MLCReshapeGradMLCReshapeGrad.gradients/concat_grad/MLCConcatV2Grad:output:3'gradients/Reshape_4_grad/Shape:output:01gradients_reshape_4_grad_mlcreshapegrad_reshape_4*
T0*
_output_shapes

:2)
'gradients/Reshape_4_grad/MLCReshapeGrad?
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
gradients/Reshape_5_grad/Shape?
'gradients/Reshape_5_grad/MLCReshapeGradMLCReshapeGrad.gradients/concat_grad/MLCConcatV2Grad:output:4'gradients/Reshape_5_grad/Shape:output:01gradients_reshape_5_grad_mlcreshapegrad_reshape_5*
T0*
_output_shapes

:2)
'gradients/Reshape_5_grad/MLCReshapeGrad?
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
gradients/Reshape_6_grad/Shape?
'gradients/Reshape_6_grad/MLCReshapeGradMLCReshapeGrad.gradients/concat_grad/MLCConcatV2Grad:output:5'gradients/Reshape_6_grad/Shape:output:01gradients_reshape_6_grad_mlcreshapegrad_reshape_6*
T0*
_output_shapes

:2)
'gradients/Reshape_6_grad/MLCReshapeGrad?
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2 
gradients/Reshape_7_grad/Shape?
'gradients/Reshape_7_grad/MLCReshapeGradMLCReshapeGrad.gradients/concat_grad/MLCConcatV2Grad:output:6'gradients/Reshape_7_grad/Shape:output:01gradients_reshape_7_grad_mlcreshapegrad_reshape_7*
T0*
_output_shapes
:2)
'gradients/Reshape_7_grad/MLCReshapeGrad?
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2 
gradients/Reshape_8_grad/Shape?
'gradients/Reshape_8_grad/MLCReshapeGradMLCReshapeGrad.gradients/concat_grad/MLCConcatV2Grad:output:7'gradients/Reshape_8_grad/Shape:output:01gradients_reshape_8_grad_mlcreshapegrad_reshape_8*
T0*
_output_shapes
:2)
'gradients/Reshape_8_grad/MLCReshapeGrad?
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2 
gradients/Reshape_9_grad/Shape?
'gradients/Reshape_9_grad/MLCReshapeGradMLCReshapeGrad.gradients/concat_grad/MLCConcatV2Grad:output:8'gradients/Reshape_9_grad/Shape:output:01gradients_reshape_9_grad_mlcreshapegrad_reshape_9*
T0*
_output_shapes
:2)
'gradients/Reshape_9_grad/MLCReshapeGrad?
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2!
gradients/Reshape_10_grad/Shape?
(gradients/Reshape_10_grad/MLCReshapeGradMLCReshapeGrad.gradients/concat_grad/MLCConcatV2Grad:output:9(gradients/Reshape_10_grad/Shape:output:03gradients_reshape_10_grad_mlcreshapegrad_reshape_10*
T0*
_output_shapes
:2*
(gradients/Reshape_10_grad/MLCReshapeGrad?
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2!
gradients/Reshape_11_grad/Shape?
(gradients/Reshape_11_grad/MLCReshapeGradMLCReshapeGrad/gradients/concat_grad/MLCConcatV2Grad:output:10(gradients/Reshape_11_grad/Shape:output:03gradients_reshape_11_grad_mlcreshapegrad_reshape_11*
T0*
_output_shapes
:2*
(gradients/Reshape_11_grad/MLCReshapeGrad?
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2!
gradients/Reshape_12_grad/Shape?
(gradients/Reshape_12_grad/MLCReshapeGradMLCReshapeGrad/gradients/concat_grad/MLCConcatV2Grad:output:11(gradients/Reshape_12_grad/Shape:output:03gradients_reshape_12_grad_mlcreshapegrad_reshape_12*
T0*
_output_shapes
:2*
(gradients/Reshape_12_grad/MLCReshapeGrad?
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:2.
,gradients/transpose_1_grad/InvertPermutation?
+gradients/transpose_1_grad/MLCTransposeGradMLCTransposeGrad0gradients/Reshape_1_grad/MLCReshapeGrad:output:00gradients/transpose_1_grad/InvertPermutation:y:07gradients_transpose_1_grad_mlctransposegrad_transpose_1*
T0*
_output_shapes

:2-
+gradients/transpose_1_grad/MLCTransposeGrad?
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:2.
,gradients/transpose_2_grad/InvertPermutation?
+gradients/transpose_2_grad/MLCTransposeGradMLCTransposeGrad0gradients/Reshape_2_grad/MLCReshapeGrad:output:00gradients/transpose_2_grad/InvertPermutation:y:07gradients_transpose_2_grad_mlctransposegrad_transpose_2*
T0*
_output_shapes

:2-
+gradients/transpose_2_grad/MLCTransposeGrad?
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:2.
,gradients/transpose_3_grad/InvertPermutation?
+gradients/transpose_3_grad/MLCTransposeGradMLCTransposeGrad0gradients/Reshape_3_grad/MLCReshapeGrad:output:00gradients/transpose_3_grad/InvertPermutation:y:07gradients_transpose_3_grad_mlctransposegrad_transpose_3*
T0*
_output_shapes

:2-
+gradients/transpose_3_grad/MLCTransposeGrad?
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:2.
,gradients/transpose_4_grad/InvertPermutation?
+gradients/transpose_4_grad/MLCTransposeGradMLCTransposeGrad0gradients/Reshape_4_grad/MLCReshapeGrad:output:00gradients/transpose_4_grad/InvertPermutation:y:07gradients_transpose_4_grad_mlctransposegrad_transpose_4*
T0*
_output_shapes

:2-
+gradients/transpose_4_grad/MLCTransposeGrad?
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:2.
,gradients/transpose_5_grad/InvertPermutation?
+gradients/transpose_5_grad/MLCTransposeGradMLCTransposeGrad0gradients/Reshape_5_grad/MLCReshapeGrad:output:00gradients/transpose_5_grad/InvertPermutation:y:07gradients_transpose_5_grad_mlctransposegrad_transpose_5*
T0*
_output_shapes

:2-
+gradients/transpose_5_grad/MLCTransposeGrad?
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:2.
,gradients/transpose_6_grad/InvertPermutation?
+gradients/transpose_6_grad/MLCTransposeGradMLCTransposeGrad0gradients/Reshape_6_grad/MLCReshapeGrad:output:00gradients/transpose_6_grad/InvertPermutation:y:07gradients_transpose_6_grad_mlctransposegrad_transpose_6*
T0*
_output_shapes

:2-
+gradients/transpose_6_grad/MLCTransposeGrad?
gradients/split_2_grad/concatConcatV20gradients/Reshape_7_grad/MLCReshapeGrad:output:00gradients/Reshape_8_grad/MLCReshapeGrad:output:00gradients/Reshape_9_grad/MLCReshapeGrad:output:01gradients/Reshape_10_grad/MLCReshapeGrad:output:01gradients/Reshape_11_grad/MLCReshapeGrad:output:01gradients/Reshape_12_grad/MLCReshapeGrad:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes	
:?2
gradients/split_2_grad/concat?
!gradients/split_grad/MLCSplitGradMLCSplitGrad/gradients/transpose_1_grad/MLCTransposeGrad:y:0/gradients/transpose_2_grad/MLCTransposeGrad:y:0/gradients/transpose_3_grad/MLCTransposeGrad:y:01gradients_split_grad_mlcsplitgrad_split_split_dim'gradients_split_grad_mlcsplitgrad_split*
N*
T0*
_output_shapes

:Z2#
!gradients/split_grad/MLCSplitGrad?
#gradients/split_1_grad/MLCSplitGradMLCSplitGrad/gradients/transpose_4_grad/MLCTransposeGrad:y:0/gradients/transpose_5_grad/MLCTransposeGrad:y:0/gradients/transpose_6_grad/MLCTransposeGrad:y:05gradients_split_1_grad_mlcsplitgrad_split_1_split_dim+gradients_split_1_grad_mlcsplitgrad_split_1*
N*
T0*
_output_shapes

:Z2%
#gradients/split_1_grad/MLCSplitGrad?
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"   Z   2
gradients/Reshape_grad/Shape?
%gradients/Reshape_grad/MLCReshapeGradMLCReshapeGrad&gradients/split_2_grad/concat:output:0%gradients/Reshape_grad/Shape:output:0-gradients_reshape_grad_mlcreshapegrad_reshape*
T0*
_output_shapes

:Z2'
%gradients/Reshape_grad/MLCReshapeGrad?
IdentityIdentity-gradients/transpose_grad/MLCTransposeGrad:y:0*
T0*,
_output_shapes
:??????????2

Identity?

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0*
T0*'
_output_shapes
:?????????2

Identity_1y

Identity_2Identity*gradients/split_grad/MLCSplitGrad:output:0*
T0*
_output_shapes

:Z2

Identity_2{

Identity_3Identity,gradients/split_1_grad/MLCSplitGrad:output:0*
T0*
_output_shapes

:Z2

Identity_3}

Identity_4Identity.gradients/Reshape_grad/MLCReshapeGrad:output:0*
T0*
_output_shapes

:Z2

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*?
_input_shapes?
?:?????????:??????????:?????????: :??????????:?????????::??????????:?????????: ::??????????:?????????: :?::?????????: :?:?:?:?:?:?::::::::::::::::::: : :: ::?*<
api_implements*(gru_4820c462-f968-4106-b392-2a4aa0828265*
api_preferred_deviceGPU*@
forward_function_name'%__forward_gpu_gru_with_fallback_16663*
go_backwards( *

time_major( :- )
'
_output_shapes
:?????????:2.
,
_output_shapes
:??????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :2.
,
_output_shapes
:??????????:-)
'
_output_shapes
:?????????: 

_output_shapes
::2.
,
_output_shapes
:??????????:1-
+
_output_shapes
:?????????:	

_output_shapes
: :


_output_shapes
::2.
,
_output_shapes
:??????????:1-
+
_output_shapes
:?????????:

_output_shapes
: :!

_output_shapes	
:?: 

_output_shapes
::-)
'
_output_shapes
:?????????:

_output_shapes
: :!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::$ 

_output_shapes

::  

_output_shapes
::$! 

_output_shapes

:: "

_output_shapes
::$# 

_output_shapes

:: $

_output_shapes
::$% 

_output_shapes

:: &

_output_shapes
::$' 

_output_shapes

:: (

_output_shapes
::$) 

_output_shapes

::*

_output_shapes
: :+

_output_shapes
: :$, 

_output_shapes

::-

_output_shapes
: :$. 

_output_shapes

::!/

_output_shapes	
:?
?2
?
while_body_13896
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_matmul_kernel_0
while_biasadd_unstack_0%
!while_matmul_1_recurrent_kernel_0
while_biasadd_1_unstack_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_matmul_kernel
while_biasadd_unstack#
while_matmul_1_recurrent_kernel
while_biasadd_1_unstack?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
while/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while_matmul_kernel_0*
T0*'
_output_shapes
:?????????Z2
while/MatMul?
while/BiasAddBiasAddwhile/MatMul:product:0while_biasadd_unstack_0*
T0*'
_output_shapes
:?????????Z2
while/BiasAdd\
while/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/Constp
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
while/split/split_dim?
while/splitSplitwhile/split/split_dim:output:0while/BiasAdd:output:0*
T0*M
_output_shapes;
9:?????????:?????????:?????????*
	num_split2
while/split?
while/MatMul_1MatMulwhile_placeholder_2!while_matmul_1_recurrent_kernel_0*
T0*'
_output_shapes
:?????????Z2
while/MatMul_1?
while/BiasAdd_1BiasAddwhile/MatMul_1:product:0while_biasadd_1_unstack_0*
T0*'
_output_shapes
:?????????Z2
while/BiasAdd_1`
while/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
while/Const_1t
while/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
while/split_1/split_dim?
while/split_1Split while/split_1/split_dim:output:0while/BiasAdd_1:output:0*
T0*M
_output_shapes;
9:?????????:?????????:?????????*
	num_split2
while/split_1
	while/addAddV2while/split:output:0while/split_1:output:0*
T0*'
_output_shapes
:?????????2
	while/addj
while/SigmoidSigmoidwhile/add:z:0*
T0*'
_output_shapes
:?????????2
while/Sigmoid?
while/add_1AddV2while/split:output:1while/split_1:output:1*
T0*'
_output_shapes
:?????????2
while/add_1p
while/Sigmoid_1Sigmoidwhile/add_1:z:0*
T0*'
_output_shapes
:?????????2
while/Sigmoid_1|
	while/mulMulwhile/Sigmoid_1:y:0while/split_1:output:2*
T0*'
_output_shapes
:?????????2
	while/mulz
while/add_2AddV2while/split:output:2while/mul:z:0*
T0*'
_output_shapes
:?????????2
while/add_2c

while/TanhTanhwhile/add_2:z:0*
T0*'
_output_shapes
:?????????2

while/Tanh{
while/mul_1Mulwhile/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:?????????2
while/mul_1_
while/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
while/sub/xx
	while/subSubwhile/sub/x:output:0while/Sigmoid:y:0*
T0*'
_output_shapes
:?????????2
	while/subr
while/mul_2Mulwhile/sub:z:0while/Tanh:y:0*
T0*'
_output_shapes
:?????????2
while/mul_2w
while/add_3AddV2while/mul_1:z:0while/mul_2:z:0*
T0*'
_output_shapes
:?????????2
while/add_3?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/add_3:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem`
while/add_4/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_4/yo
while/add_4AddV2while_placeholderwhile/add_4/y:output:0*
T0*
_output_shapes
: 2
while/add_4`
while/add_5/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_5/yv
while/add_5AddV2while_while_loop_counterwhile/add_5/y:output:0*
T0*
_output_shapes
: 2
while/add_5^
while/IdentityIdentitywhile/add_5:z:0*
T0*
_output_shapes
: 2
while/Identityq
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: 2
while/Identity_1b
while/Identity_2Identitywhile/add_4:z:0*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3s
while/Identity_4Identitywhile/add_3:z:0*
T0*'
_output_shapes
:?????????2
while/Identity_4"4
while_biasadd_1_unstackwhile_biasadd_1_unstack_0"0
while_biasadd_unstackwhile_biasadd_unstack_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"D
while_matmul_1_recurrent_kernel!while_matmul_1_recurrent_kernel_0",
while_matmul_kernelwhile_matmul_kernel_0",
while_strided_slicewhile_strided_slice_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*R
_input_shapesA
?: : : : :?????????: : :Z:Z:Z:Z: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:Z: 

_output_shapes
:Z:$	 

_output_shapes

:Z: 


_output_shapes
:Z
ڮ
?
8__inference___backward_gpu_gru_with_fallback_18997_19215
placeholder
placeholder_1
placeholder_2
placeholder_3/
+gradients_strided_slice_grad_shape_cudnnrnnB
>gradients_strided_slice_grad_mlcstridedslicegrad_strided_sliceA
=gradients_transpose_7_grad_invertpermutation_transpose_7_perm;
7gradients_transpose_7_grad_mlctransposegrad_transpose_7)
%gradients_squeeze_grad_shape_cudnnrnn!
gradients_zeros_like_cudnnrnn#
gradients_zeros_like_1_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims=
9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c3
/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h)
%gradients_concat_grad_add_concat_axis5
1gradients_reshape_1_grad_mlcreshapegrad_reshape_15
1gradients_reshape_2_grad_mlcreshapegrad_reshape_25
1gradients_reshape_3_grad_mlcreshapegrad_reshape_35
1gradients_reshape_4_grad_mlcreshapegrad_reshape_45
1gradients_reshape_5_grad_mlcreshapegrad_reshape_55
1gradients_reshape_6_grad_mlcreshapegrad_reshape_65
1gradients_reshape_7_grad_mlcreshapegrad_reshape_75
1gradients_reshape_8_grad_mlcreshapegrad_reshape_85
1gradients_reshape_9_grad_mlcreshapegrad_reshape_97
3gradients_reshape_10_grad_mlcreshapegrad_reshape_107
3gradients_reshape_11_grad_mlcreshapegrad_reshape_117
3gradients_reshape_12_grad_mlcreshapegrad_reshape_12A
=gradients_transpose_1_grad_invertpermutation_transpose_1_perm;
7gradients_transpose_1_grad_mlctransposegrad_transpose_1A
=gradients_transpose_2_grad_invertpermutation_transpose_2_perm;
7gradients_transpose_2_grad_mlctransposegrad_transpose_2A
=gradients_transpose_3_grad_invertpermutation_transpose_3_perm;
7gradients_transpose_3_grad_mlctransposegrad_transpose_3A
=gradients_transpose_4_grad_invertpermutation_transpose_4_perm;
7gradients_transpose_4_grad_mlctransposegrad_transpose_4A
=gradients_transpose_5_grad_invertpermutation_transpose_5_perm;
7gradients_transpose_5_grad_mlctransposegrad_transpose_5A
=gradients_transpose_6_grad_invertpermutation_transpose_6_perm;
7gradients_transpose_6_grad_mlctransposegrad_transpose_63
/gradients_split_2_grad_concat_split_2_split_dim5
1gradients_split_grad_mlcsplitgrad_split_split_dim+
'gradients_split_grad_mlcsplitgrad_split9
5gradients_split_1_grad_mlcsplitgrad_split_1_split_dim/
+gradients_split_1_grad_mlcsplitgrad_split_11
-gradients_reshape_grad_mlcreshapegrad_reshape
identity

identity_1

identity_2

identity_3

identity_4?u
gradients/grad_ys_0Identityplaceholder*
T0*'
_output_shapes
:?????????2
gradients/grad_ys_0?
gradients/grad_ys_1Identityplaceholder_1*
T0*4
_output_shapes"
 :??????????????????2
gradients/grad_ys_1w
gradients/grad_ys_2Identityplaceholder_2*
T0*'
_output_shapes
:?????????2
gradients/grad_ys_2f
gradients/grad_ys_3Identityplaceholder_3*
T0*
_output_shapes
: 2
gradients/grad_ys_3?
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:2$
"gradients/strided_slice_grad/Shape?
6gradients/strided_slice_grad/MLCStridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*
valueB:
?????????28
6gradients/strided_slice_grad/MLCStridedSliceGrad/begin?
4gradients/strided_slice_grad/MLCStridedSliceGrad/endConst*
_output_shapes
:*
dtype0*
valueB: 26
4gradients/strided_slice_grad/MLCStridedSliceGrad/end?
8gradients/strided_slice_grad/MLCStridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*
valueB:2:
8gradients/strided_slice_grad/MLCStridedSliceGrad/strides?
0gradients/strided_slice_grad/MLCStridedSliceGradMLCStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0?gradients/strided_slice_grad/MLCStridedSliceGrad/begin:output:0=gradients/strided_slice_grad/MLCStridedSliceGrad/end:output:0Agradients/strided_slice_grad/MLCStridedSliceGrad/strides:output:0gradients/grad_ys_0:output:0>gradients_strided_slice_grad_mlcstridedslicegrad_strided_slice*
Index0*
T0*4
_output_shapes"
 :??????????????????*
shrink_axis_mask22
0gradients/strided_slice_grad/MLCStridedSliceGrad?
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:2.
,gradients/transpose_7_grad/InvertPermutation?
+gradients/transpose_7_grad/MLCTransposeGradMLCTransposeGradgradients/grad_ys_1:output:00gradients/transpose_7_grad/InvertPermutation:y:07gradients_transpose_7_grad_mlctransposegrad_transpose_7*
T0*4
_output_shapes"
 :??????????????????2-
+gradients/transpose_7_grad/MLCTransposeGrad?
gradients/Squeeze_grad/ShapeShape%gradients_squeeze_grad_shape_cudnnrnn*
T0*
_output_shapes
:2
gradients/Squeeze_grad/Shape?
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*+
_output_shapes
:?????????2 
gradients/Squeeze_grad/Reshape?
gradients/AddNAddN9gradients/strided_slice_grad/MLCStridedSliceGrad:output:0/gradients/transpose_7_grad/MLCTransposeGrad:y:0*
N*
T0*C
_class9
75loc:@gradients/strided_slice_grad/MLCStridedSliceGrad*4
_output_shapes"
 :??????????????????2
gradients/AddNy
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
: 2
gradients/zeros_like?
gradients/zeros_like_1	ZerosLikegradients_zeros_like_1_cudnnrnn*
T0*
_output_shapes
:2
gradients/zeros_like_1?
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat+gradients_strided_slice_grad_shape_cudnnrnn%gradients_squeeze_grad_shape_cudnnrnngradients_zeros_like_cudnnrnngradients/AddN:sum:0'gradients/Squeeze_grad/Reshape:output:0gradients/zeros_like:y:0gradients_zeros_like_1_cudnnrnn*
T0*T
_output_shapesB
@:??????????????????:?????????: :?*
rnn_modegru2*
(gradients/CudnnRNN_grad/CudnnRNNBackprop?
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:2,
*gradients/transpose_grad/InvertPermutation?
)gradients/transpose_grad/MLCTransposeGradMLCTransposeGrad9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:02gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose*
T0*4
_output_shapes"
 :??????????????????2+
)gradients/transpose_grad/MLCTransposeGrad?
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
:2!
gradients/ExpandDims_grad/Shape?
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*'
_output_shapes
:?????????2#
!gradients/ExpandDims_grad/Reshape?
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:?2
gradients/concat_grad/Shape|
gradients/concat_grad/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_grad/add/y?
gradients/concat_grad/addAddV2%gradients_concat_grad_add_concat_axis$gradients/concat_grad/add/y:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/add?
)gradients/concat_grad/strided_slice/stackPack%gradients_concat_grad_add_concat_axis*
N*
T0*
_output_shapes
:2+
)gradients/concat_grad/strided_slice/stack?
+gradients/concat_grad/strided_slice/stack_1Packgradients/concat_grad/add:z:0*
N*
T0*
_output_shapes
:2-
+gradients/concat_grad/strided_slice/stack_1?
+gradients/concat_grad/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+gradients/concat_grad/strided_slice/stack_2?
#gradients/concat_grad/strided_sliceStridedSlice$gradients/concat_grad/Shape:output:02gradients/concat_grad/strided_slice/stack:output:04gradients/concat_grad/strided_slice/stack_1:output:04gradients/concat_grad/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#gradients/concat_grad/strided_slice?
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:?2
gradients/concat_grad/Shape_1?
gradients/concat_grad/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_grad/add_1/y?
gradients/concat_grad/add_1AddV2%gradients_concat_grad_add_concat_axis&gradients/concat_grad/add_1/y:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/add_1?
+gradients/concat_grad/strided_slice_1/stackPack%gradients_concat_grad_add_concat_axis*
N*
T0*
_output_shapes
:2-
+gradients/concat_grad/strided_slice_1/stack?
-gradients/concat_grad/strided_slice_1/stack_1Packgradients/concat_grad/add_1:z:0*
N*
T0*
_output_shapes
:2/
-gradients/concat_grad/strided_slice_1/stack_1?
-gradients/concat_grad/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-gradients/concat_grad/strided_slice_1/stack_2?
%gradients/concat_grad/strided_slice_1StridedSlice&gradients/concat_grad/Shape_1:output:04gradients/concat_grad/strided_slice_1/stack:output:06gradients/concat_grad/strided_slice_1/stack_1:output:06gradients/concat_grad/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%gradients/concat_grad/strided_slice_1?
gradients/concat_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:?2
gradients/concat_grad/Shape_2?
gradients/concat_grad/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_grad/add_2/y?
gradients/concat_grad/add_2AddV2%gradients_concat_grad_add_concat_axis&gradients/concat_grad/add_2/y:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/add_2?
+gradients/concat_grad/strided_slice_2/stackPack%gradients_concat_grad_add_concat_axis*
N*
T0*
_output_shapes
:2-
+gradients/concat_grad/strided_slice_2/stack?
-gradients/concat_grad/strided_slice_2/stack_1Packgradients/concat_grad/add_2:z:0*
N*
T0*
_output_shapes
:2/
-gradients/concat_grad/strided_slice_2/stack_1?
-gradients/concat_grad/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-gradients/concat_grad/strided_slice_2/stack_2?
%gradients/concat_grad/strided_slice_2StridedSlice&gradients/concat_grad/Shape_2:output:04gradients/concat_grad/strided_slice_2/stack:output:06gradients/concat_grad/strided_slice_2/stack_1:output:06gradients/concat_grad/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%gradients/concat_grad/strided_slice_2?
gradients/concat_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:?2
gradients/concat_grad/Shape_3?
gradients/concat_grad/add_3/yConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_grad/add_3/y?
gradients/concat_grad/add_3AddV2%gradients_concat_grad_add_concat_axis&gradients/concat_grad/add_3/y:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/add_3?
+gradients/concat_grad/strided_slice_3/stackPack%gradients_concat_grad_add_concat_axis*
N*
T0*
_output_shapes
:2-
+gradients/concat_grad/strided_slice_3/stack?
-gradients/concat_grad/strided_slice_3/stack_1Packgradients/concat_grad/add_3:z:0*
N*
T0*
_output_shapes
:2/
-gradients/concat_grad/strided_slice_3/stack_1?
-gradients/concat_grad/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-gradients/concat_grad/strided_slice_3/stack_2?
%gradients/concat_grad/strided_slice_3StridedSlice&gradients/concat_grad/Shape_3:output:04gradients/concat_grad/strided_slice_3/stack:output:06gradients/concat_grad/strided_slice_3/stack_1:output:06gradients/concat_grad/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%gradients/concat_grad/strided_slice_3?
gradients/concat_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:?2
gradients/concat_grad/Shape_4?
gradients/concat_grad/add_4/yConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_grad/add_4/y?
gradients/concat_grad/add_4AddV2%gradients_concat_grad_add_concat_axis&gradients/concat_grad/add_4/y:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/add_4?
+gradients/concat_grad/strided_slice_4/stackPack%gradients_concat_grad_add_concat_axis*
N*
T0*
_output_shapes
:2-
+gradients/concat_grad/strided_slice_4/stack?
-gradients/concat_grad/strided_slice_4/stack_1Packgradients/concat_grad/add_4:z:0*
N*
T0*
_output_shapes
:2/
-gradients/concat_grad/strided_slice_4/stack_1?
-gradients/concat_grad/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-gradients/concat_grad/strided_slice_4/stack_2?
%gradients/concat_grad/strided_slice_4StridedSlice&gradients/concat_grad/Shape_4:output:04gradients/concat_grad/strided_slice_4/stack:output:06gradients/concat_grad/strided_slice_4/stack_1:output:06gradients/concat_grad/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%gradients/concat_grad/strided_slice_4?
gradients/concat_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:?2
gradients/concat_grad/Shape_5?
gradients/concat_grad/add_5/yConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_grad/add_5/y?
gradients/concat_grad/add_5AddV2%gradients_concat_grad_add_concat_axis&gradients/concat_grad/add_5/y:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/add_5?
+gradients/concat_grad/strided_slice_5/stackPack%gradients_concat_grad_add_concat_axis*
N*
T0*
_output_shapes
:2-
+gradients/concat_grad/strided_slice_5/stack?
-gradients/concat_grad/strided_slice_5/stack_1Packgradients/concat_grad/add_5:z:0*
N*
T0*
_output_shapes
:2/
-gradients/concat_grad/strided_slice_5/stack_1?
-gradients/concat_grad/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-gradients/concat_grad/strided_slice_5/stack_2?
%gradients/concat_grad/strided_slice_5StridedSlice&gradients/concat_grad/Shape_5:output:04gradients/concat_grad/strided_slice_5/stack:output:06gradients/concat_grad/strided_slice_5/stack_1:output:06gradients/concat_grad/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%gradients/concat_grad/strided_slice_5?
gradients/concat_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape_6?
gradients/concat_grad/add_6/yConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_grad/add_6/y?
gradients/concat_grad/add_6AddV2%gradients_concat_grad_add_concat_axis&gradients/concat_grad/add_6/y:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/add_6?
+gradients/concat_grad/strided_slice_6/stackPack%gradients_concat_grad_add_concat_axis*
N*
T0*
_output_shapes
:2-
+gradients/concat_grad/strided_slice_6/stack?
-gradients/concat_grad/strided_slice_6/stack_1Packgradients/concat_grad/add_6:z:0*
N*
T0*
_output_shapes
:2/
-gradients/concat_grad/strided_slice_6/stack_1?
-gradients/concat_grad/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-gradients/concat_grad/strided_slice_6/stack_2?
%gradients/concat_grad/strided_slice_6StridedSlice&gradients/concat_grad/Shape_6:output:04gradients/concat_grad/strided_slice_6/stack:output:06gradients/concat_grad/strided_slice_6/stack_1:output:06gradients/concat_grad/strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%gradients/concat_grad/strided_slice_6?
gradients/concat_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape_7?
gradients/concat_grad/add_7/yConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_grad/add_7/y?
gradients/concat_grad/add_7AddV2%gradients_concat_grad_add_concat_axis&gradients/concat_grad/add_7/y:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/add_7?
+gradients/concat_grad/strided_slice_7/stackPack%gradients_concat_grad_add_concat_axis*
N*
T0*
_output_shapes
:2-
+gradients/concat_grad/strided_slice_7/stack?
-gradients/concat_grad/strided_slice_7/stack_1Packgradients/concat_grad/add_7:z:0*
N*
T0*
_output_shapes
:2/
-gradients/concat_grad/strided_slice_7/stack_1?
-gradients/concat_grad/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-gradients/concat_grad/strided_slice_7/stack_2?
%gradients/concat_grad/strided_slice_7StridedSlice&gradients/concat_grad/Shape_7:output:04gradients/concat_grad/strided_slice_7/stack:output:06gradients/concat_grad/strided_slice_7/stack_1:output:06gradients/concat_grad/strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%gradients/concat_grad/strided_slice_7?
gradients/concat_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape_8?
gradients/concat_grad/add_8/yConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_grad/add_8/y?
gradients/concat_grad/add_8AddV2%gradients_concat_grad_add_concat_axis&gradients/concat_grad/add_8/y:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/add_8?
+gradients/concat_grad/strided_slice_8/stackPack%gradients_concat_grad_add_concat_axis*
N*
T0*
_output_shapes
:2-
+gradients/concat_grad/strided_slice_8/stack?
-gradients/concat_grad/strided_slice_8/stack_1Packgradients/concat_grad/add_8:z:0*
N*
T0*
_output_shapes
:2/
-gradients/concat_grad/strided_slice_8/stack_1?
-gradients/concat_grad/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-gradients/concat_grad/strided_slice_8/stack_2?
%gradients/concat_grad/strided_slice_8StridedSlice&gradients/concat_grad/Shape_8:output:04gradients/concat_grad/strided_slice_8/stack:output:06gradients/concat_grad/strided_slice_8/stack_1:output:06gradients/concat_grad/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%gradients/concat_grad/strided_slice_8?
gradients/concat_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape_9?
gradients/concat_grad/add_9/yConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_grad/add_9/y?
gradients/concat_grad/add_9AddV2%gradients_concat_grad_add_concat_axis&gradients/concat_grad/add_9/y:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/add_9?
+gradients/concat_grad/strided_slice_9/stackPack%gradients_concat_grad_add_concat_axis*
N*
T0*
_output_shapes
:2-
+gradients/concat_grad/strided_slice_9/stack?
-gradients/concat_grad/strided_slice_9/stack_1Packgradients/concat_grad/add_9:z:0*
N*
T0*
_output_shapes
:2/
-gradients/concat_grad/strided_slice_9/stack_1?
-gradients/concat_grad/strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-gradients/concat_grad/strided_slice_9/stack_2?
%gradients/concat_grad/strided_slice_9StridedSlice&gradients/concat_grad/Shape_9:output:04gradients/concat_grad/strided_slice_9/stack:output:06gradients/concat_grad/strided_slice_9/stack_1:output:06gradients/concat_grad/strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%gradients/concat_grad/strided_slice_9?
gradients/concat_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB:2 
gradients/concat_grad/Shape_10?
gradients/concat_grad/add_10/yConst*
_output_shapes
: *
dtype0*
value	B :2 
gradients/concat_grad/add_10/y?
gradients/concat_grad/add_10AddV2%gradients_concat_grad_add_concat_axis'gradients/concat_grad/add_10/y:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/add_10?
,gradients/concat_grad/strided_slice_10/stackPack%gradients_concat_grad_add_concat_axis*
N*
T0*
_output_shapes
:2.
,gradients/concat_grad/strided_slice_10/stack?
.gradients/concat_grad/strided_slice_10/stack_1Pack gradients/concat_grad/add_10:z:0*
N*
T0*
_output_shapes
:20
.gradients/concat_grad/strided_slice_10/stack_1?
.gradients/concat_grad/strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.gradients/concat_grad/strided_slice_10/stack_2?
&gradients/concat_grad/strided_slice_10StridedSlice'gradients/concat_grad/Shape_10:output:05gradients/concat_grad/strided_slice_10/stack:output:07gradients/concat_grad/strided_slice_10/stack_1:output:07gradients/concat_grad/strided_slice_10/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2(
&gradients/concat_grad/strided_slice_10?
gradients/concat_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB:2 
gradients/concat_grad/Shape_11?
gradients/concat_grad/add_11/yConst*
_output_shapes
: *
dtype0*
value	B :2 
gradients/concat_grad/add_11/y?
gradients/concat_grad/add_11AddV2%gradients_concat_grad_add_concat_axis'gradients/concat_grad/add_11/y:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/add_11?
,gradients/concat_grad/strided_slice_11/stackPack%gradients_concat_grad_add_concat_axis*
N*
T0*
_output_shapes
:2.
,gradients/concat_grad/strided_slice_11/stack?
.gradients/concat_grad/strided_slice_11/stack_1Pack gradients/concat_grad/add_11:z:0*
N*
T0*
_output_shapes
:20
.gradients/concat_grad/strided_slice_11/stack_1?
.gradients/concat_grad/strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.gradients/concat_grad/strided_slice_11/stack_2?
&gradients/concat_grad/strided_slice_11StridedSlice'gradients/concat_grad/Shape_11:output:05gradients/concat_grad/strided_slice_11/stack:output:07gradients/concat_grad/strided_slice_11/stack_1:output:07gradients/concat_grad/strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2(
&gradients/concat_grad/strided_slice_11?
gradients/concat_grad/packedPack,gradients/concat_grad/strided_slice:output:0.gradients/concat_grad/strided_slice_1:output:0.gradients/concat_grad/strided_slice_2:output:0.gradients/concat_grad/strided_slice_3:output:0.gradients/concat_grad/strided_slice_4:output:0.gradients/concat_grad/strided_slice_5:output:0.gradients/concat_grad/strided_slice_6:output:0.gradients/concat_grad/strided_slice_7:output:0.gradients/concat_grad/strided_slice_8:output:0.gradients/concat_grad/strided_slice_9:output:0/gradients/concat_grad/strided_slice_10:output:0/gradients/concat_grad/strided_slice_11:output:0*
N*
T0*
_output_shapes
:2
gradients/concat_grad/packed?
%gradients/concat_grad/MLCConcatV2GradMLCConcatV2Grad:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0%gradients/concat_grad/packed:output:0%gradients_concat_grad_add_concat_axis$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0&gradients/concat_grad/Shape_2:output:0&gradients/concat_grad/Shape_3:output:0&gradients/concat_grad/Shape_4:output:0&gradients/concat_grad/Shape_5:output:0&gradients/concat_grad/Shape_6:output:0&gradients/concat_grad/Shape_7:output:0&gradients/concat_grad/Shape_8:output:0&gradients/concat_grad/Shape_9:output:0'gradients/concat_grad/Shape_10:output:0'gradients/concat_grad/Shape_11:output:0/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat*
N*
T0*

Tlen0*b
_output_shapesP
N:?:?:?:?:?:?::::::2'
%gradients/concat_grad/MLCConcatV2Grad?
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
gradients/Reshape_1_grad/Shape?
'gradients/Reshape_1_grad/MLCReshapeGradMLCReshapeGrad.gradients/concat_grad/MLCConcatV2Grad:output:0'gradients/Reshape_1_grad/Shape:output:01gradients_reshape_1_grad_mlcreshapegrad_reshape_1*
T0*
_output_shapes

:2)
'gradients/Reshape_1_grad/MLCReshapeGrad?
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
gradients/Reshape_2_grad/Shape?
'gradients/Reshape_2_grad/MLCReshapeGradMLCReshapeGrad.gradients/concat_grad/MLCConcatV2Grad:output:1'gradients/Reshape_2_grad/Shape:output:01gradients_reshape_2_grad_mlcreshapegrad_reshape_2*
T0*
_output_shapes

:2)
'gradients/Reshape_2_grad/MLCReshapeGrad?
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
gradients/Reshape_3_grad/Shape?
'gradients/Reshape_3_grad/MLCReshapeGradMLCReshapeGrad.gradients/concat_grad/MLCConcatV2Grad:output:2'gradients/Reshape_3_grad/Shape:output:01gradients_reshape_3_grad_mlcreshapegrad_reshape_3*
T0*
_output_shapes

:2)
'gradients/Reshape_3_grad/MLCReshapeGrad?
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
gradients/Reshape_4_grad/Shape?
'gradients/Reshape_4_grad/MLCReshapeGradMLCReshapeGrad.gradients/concat_grad/MLCConcatV2Grad:output:3'gradients/Reshape_4_grad/Shape:output:01gradients_reshape_4_grad_mlcreshapegrad_reshape_4*
T0*
_output_shapes

:2)
'gradients/Reshape_4_grad/MLCReshapeGrad?
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
gradients/Reshape_5_grad/Shape?
'gradients/Reshape_5_grad/MLCReshapeGradMLCReshapeGrad.gradients/concat_grad/MLCConcatV2Grad:output:4'gradients/Reshape_5_grad/Shape:output:01gradients_reshape_5_grad_mlcreshapegrad_reshape_5*
T0*
_output_shapes

:2)
'gradients/Reshape_5_grad/MLCReshapeGrad?
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
gradients/Reshape_6_grad/Shape?
'gradients/Reshape_6_grad/MLCReshapeGradMLCReshapeGrad.gradients/concat_grad/MLCConcatV2Grad:output:5'gradients/Reshape_6_grad/Shape:output:01gradients_reshape_6_grad_mlcreshapegrad_reshape_6*
T0*
_output_shapes

:2)
'gradients/Reshape_6_grad/MLCReshapeGrad?
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2 
gradients/Reshape_7_grad/Shape?
'gradients/Reshape_7_grad/MLCReshapeGradMLCReshapeGrad.gradients/concat_grad/MLCConcatV2Grad:output:6'gradients/Reshape_7_grad/Shape:output:01gradients_reshape_7_grad_mlcreshapegrad_reshape_7*
T0*
_output_shapes
:2)
'gradients/Reshape_7_grad/MLCReshapeGrad?
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2 
gradients/Reshape_8_grad/Shape?
'gradients/Reshape_8_grad/MLCReshapeGradMLCReshapeGrad.gradients/concat_grad/MLCConcatV2Grad:output:7'gradients/Reshape_8_grad/Shape:output:01gradients_reshape_8_grad_mlcreshapegrad_reshape_8*
T0*
_output_shapes
:2)
'gradients/Reshape_8_grad/MLCReshapeGrad?
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2 
gradients/Reshape_9_grad/Shape?
'gradients/Reshape_9_grad/MLCReshapeGradMLCReshapeGrad.gradients/concat_grad/MLCConcatV2Grad:output:8'gradients/Reshape_9_grad/Shape:output:01gradients_reshape_9_grad_mlcreshapegrad_reshape_9*
T0*
_output_shapes
:2)
'gradients/Reshape_9_grad/MLCReshapeGrad?
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2!
gradients/Reshape_10_grad/Shape?
(gradients/Reshape_10_grad/MLCReshapeGradMLCReshapeGrad.gradients/concat_grad/MLCConcatV2Grad:output:9(gradients/Reshape_10_grad/Shape:output:03gradients_reshape_10_grad_mlcreshapegrad_reshape_10*
T0*
_output_shapes
:2*
(gradients/Reshape_10_grad/MLCReshapeGrad?
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2!
gradients/Reshape_11_grad/Shape?
(gradients/Reshape_11_grad/MLCReshapeGradMLCReshapeGrad/gradients/concat_grad/MLCConcatV2Grad:output:10(gradients/Reshape_11_grad/Shape:output:03gradients_reshape_11_grad_mlcreshapegrad_reshape_11*
T0*
_output_shapes
:2*
(gradients/Reshape_11_grad/MLCReshapeGrad?
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2!
gradients/Reshape_12_grad/Shape?
(gradients/Reshape_12_grad/MLCReshapeGradMLCReshapeGrad/gradients/concat_grad/MLCConcatV2Grad:output:11(gradients/Reshape_12_grad/Shape:output:03gradients_reshape_12_grad_mlcreshapegrad_reshape_12*
T0*
_output_shapes
:2*
(gradients/Reshape_12_grad/MLCReshapeGrad?
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:2.
,gradients/transpose_1_grad/InvertPermutation?
+gradients/transpose_1_grad/MLCTransposeGradMLCTransposeGrad0gradients/Reshape_1_grad/MLCReshapeGrad:output:00gradients/transpose_1_grad/InvertPermutation:y:07gradients_transpose_1_grad_mlctransposegrad_transpose_1*
T0*
_output_shapes

:2-
+gradients/transpose_1_grad/MLCTransposeGrad?
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:2.
,gradients/transpose_2_grad/InvertPermutation?
+gradients/transpose_2_grad/MLCTransposeGradMLCTransposeGrad0gradients/Reshape_2_grad/MLCReshapeGrad:output:00gradients/transpose_2_grad/InvertPermutation:y:07gradients_transpose_2_grad_mlctransposegrad_transpose_2*
T0*
_output_shapes

:2-
+gradients/transpose_2_grad/MLCTransposeGrad?
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:2.
,gradients/transpose_3_grad/InvertPermutation?
+gradients/transpose_3_grad/MLCTransposeGradMLCTransposeGrad0gradients/Reshape_3_grad/MLCReshapeGrad:output:00gradients/transpose_3_grad/InvertPermutation:y:07gradients_transpose_3_grad_mlctransposegrad_transpose_3*
T0*
_output_shapes

:2-
+gradients/transpose_3_grad/MLCTransposeGrad?
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:2.
,gradients/transpose_4_grad/InvertPermutation?
+gradients/transpose_4_grad/MLCTransposeGradMLCTransposeGrad0gradients/Reshape_4_grad/MLCReshapeGrad:output:00gradients/transpose_4_grad/InvertPermutation:y:07gradients_transpose_4_grad_mlctransposegrad_transpose_4*
T0*
_output_shapes

:2-
+gradients/transpose_4_grad/MLCTransposeGrad?
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:2.
,gradients/transpose_5_grad/InvertPermutation?
+gradients/transpose_5_grad/MLCTransposeGradMLCTransposeGrad0gradients/Reshape_5_grad/MLCReshapeGrad:output:00gradients/transpose_5_grad/InvertPermutation:y:07gradients_transpose_5_grad_mlctransposegrad_transpose_5*
T0*
_output_shapes

:2-
+gradients/transpose_5_grad/MLCTransposeGrad?
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:2.
,gradients/transpose_6_grad/InvertPermutation?
+gradients/transpose_6_grad/MLCTransposeGradMLCTransposeGrad0gradients/Reshape_6_grad/MLCReshapeGrad:output:00gradients/transpose_6_grad/InvertPermutation:y:07gradients_transpose_6_grad_mlctransposegrad_transpose_6*
T0*
_output_shapes

:2-
+gradients/transpose_6_grad/MLCTransposeGrad?
gradients/split_2_grad/concatConcatV20gradients/Reshape_7_grad/MLCReshapeGrad:output:00gradients/Reshape_8_grad/MLCReshapeGrad:output:00gradients/Reshape_9_grad/MLCReshapeGrad:output:01gradients/Reshape_10_grad/MLCReshapeGrad:output:01gradients/Reshape_11_grad/MLCReshapeGrad:output:01gradients/Reshape_12_grad/MLCReshapeGrad:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes	
:?2
gradients/split_2_grad/concat?
!gradients/split_grad/MLCSplitGradMLCSplitGrad/gradients/transpose_1_grad/MLCTransposeGrad:y:0/gradients/transpose_2_grad/MLCTransposeGrad:y:0/gradients/transpose_3_grad/MLCTransposeGrad:y:01gradients_split_grad_mlcsplitgrad_split_split_dim'gradients_split_grad_mlcsplitgrad_split*
N*
T0*
_output_shapes

:Z2#
!gradients/split_grad/MLCSplitGrad?
#gradients/split_1_grad/MLCSplitGradMLCSplitGrad/gradients/transpose_4_grad/MLCTransposeGrad:y:0/gradients/transpose_5_grad/MLCTransposeGrad:y:0/gradients/transpose_6_grad/MLCTransposeGrad:y:05gradients_split_1_grad_mlcsplitgrad_split_1_split_dim+gradients_split_1_grad_mlcsplitgrad_split_1*
N*
T0*
_output_shapes

:Z2%
#gradients/split_1_grad/MLCSplitGrad?
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"   Z   2
gradients/Reshape_grad/Shape?
%gradients/Reshape_grad/MLCReshapeGradMLCReshapeGrad&gradients/split_2_grad/concat:output:0%gradients/Reshape_grad/Shape:output:0-gradients_reshape_grad_mlcreshapegrad_reshape*
T0*
_output_shapes

:Z2'
%gradients/Reshape_grad/MLCReshapeGrad?
IdentityIdentity-gradients/transpose_grad/MLCTransposeGrad:y:0*
T0*4
_output_shapes"
 :??????????????????2

Identity?

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0*
T0*'
_output_shapes
:?????????2

Identity_1y

Identity_2Identity*gradients/split_grad/MLCSplitGrad:output:0*
T0*
_output_shapes

:Z2

Identity_2{

Identity_3Identity,gradients/split_1_grad/MLCSplitGrad:output:0*
T0*
_output_shapes

:Z2

Identity_3}

Identity_4Identity.gradients/Reshape_grad/MLCReshapeGrad:output:0*
T0*
_output_shapes

:Z2

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*?
_input_shapes?
?:?????????:??????????????????:?????????: :??????????????????:?????????::??????????????????:?????????: ::??????????????????:?????????: :?::?????????: :?:?:?:?:?:?::::::::::::::::::: : :: ::?*<
api_implements*(gru_d29f5daa-115b-49e1-809f-181a86e00a5d*
api_preferred_deviceGPU*@
forward_function_name'%__forward_gpu_gru_with_fallback_19214*
go_backwards( *

time_major( :- )
'
_output_shapes
:?????????::6
4
_output_shapes"
 :??????????????????:-)
'
_output_shapes
:?????????:

_output_shapes
: ::6
4
_output_shapes"
 :??????????????????:-)
'
_output_shapes
:?????????: 

_output_shapes
:::6
4
_output_shapes"
 :??????????????????:1-
+
_output_shapes
:?????????:	

_output_shapes
: :


_output_shapes
:::6
4
_output_shapes"
 :??????????????????:1-
+
_output_shapes
:?????????:

_output_shapes
: :!

_output_shapes	
:?: 

_output_shapes
::-)
'
_output_shapes
:?????????:

_output_shapes
: :!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::$ 

_output_shapes

::  

_output_shapes
::$! 

_output_shapes

:: "

_output_shapes
::$# 

_output_shapes

:: $

_output_shapes
::$% 

_output_shapes

:: &

_output_shapes
::$' 

_output_shapes

:: (

_output_shapes
::$) 

_output_shapes

::*

_output_shapes
: :+

_output_shapes
: :$, 

_output_shapes

::-

_output_shapes
: :$. 

_output_shapes

::!/

_output_shapes	
:?
?
?
,__inference_sequential_1_layer_call_fn_17272
	gru_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCall	gru_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_172592
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????:::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
,
_output_shapes
:??????????
#
_user_specified_name	gru_input
??
?
8__inference___backward_gpu_gru_with_fallback_16915_17133
placeholder
placeholder_1
placeholder_2
placeholder_3/
+gradients_strided_slice_grad_shape_cudnnrnnB
>gradients_strided_slice_grad_mlcstridedslicegrad_strided_sliceA
=gradients_transpose_7_grad_invertpermutation_transpose_7_perm;
7gradients_transpose_7_grad_mlctransposegrad_transpose_7)
%gradients_squeeze_grad_shape_cudnnrnn!
gradients_zeros_like_cudnnrnn#
gradients_zeros_like_1_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims=
9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c3
/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h)
%gradients_concat_grad_add_concat_axis5
1gradients_reshape_1_grad_mlcreshapegrad_reshape_15
1gradients_reshape_2_grad_mlcreshapegrad_reshape_25
1gradients_reshape_3_grad_mlcreshapegrad_reshape_35
1gradients_reshape_4_grad_mlcreshapegrad_reshape_45
1gradients_reshape_5_grad_mlcreshapegrad_reshape_55
1gradients_reshape_6_grad_mlcreshapegrad_reshape_65
1gradients_reshape_7_grad_mlcreshapegrad_reshape_75
1gradients_reshape_8_grad_mlcreshapegrad_reshape_85
1gradients_reshape_9_grad_mlcreshapegrad_reshape_97
3gradients_reshape_10_grad_mlcreshapegrad_reshape_107
3gradients_reshape_11_grad_mlcreshapegrad_reshape_117
3gradients_reshape_12_grad_mlcreshapegrad_reshape_12A
=gradients_transpose_1_grad_invertpermutation_transpose_1_perm;
7gradients_transpose_1_grad_mlctransposegrad_transpose_1A
=gradients_transpose_2_grad_invertpermutation_transpose_2_perm;
7gradients_transpose_2_grad_mlctransposegrad_transpose_2A
=gradients_transpose_3_grad_invertpermutation_transpose_3_perm;
7gradients_transpose_3_grad_mlctransposegrad_transpose_3A
=gradients_transpose_4_grad_invertpermutation_transpose_4_perm;
7gradients_transpose_4_grad_mlctransposegrad_transpose_4A
=gradients_transpose_5_grad_invertpermutation_transpose_5_perm;
7gradients_transpose_5_grad_mlctransposegrad_transpose_5A
=gradients_transpose_6_grad_invertpermutation_transpose_6_perm;
7gradients_transpose_6_grad_mlctransposegrad_transpose_63
/gradients_split_2_grad_concat_split_2_split_dim5
1gradients_split_grad_mlcsplitgrad_split_split_dim+
'gradients_split_grad_mlcsplitgrad_split9
5gradients_split_1_grad_mlcsplitgrad_split_1_split_dim/
+gradients_split_1_grad_mlcsplitgrad_split_11
-gradients_reshape_grad_mlcreshapegrad_reshape
identity

identity_1

identity_2

identity_3

identity_4?u
gradients/grad_ys_0Identityplaceholder*
T0*'
_output_shapes
:?????????2
gradients/grad_ys_0|
gradients/grad_ys_1Identityplaceholder_1*
T0*,
_output_shapes
:??????????2
gradients/grad_ys_1w
gradients/grad_ys_2Identityplaceholder_2*
T0*'
_output_shapes
:?????????2
gradients/grad_ys_2f
gradients/grad_ys_3Identityplaceholder_3*
T0*
_output_shapes
: 2
gradients/grad_ys_3?
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:2$
"gradients/strided_slice_grad/Shape?
6gradients/strided_slice_grad/MLCStridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*
valueB:
?????????28
6gradients/strided_slice_grad/MLCStridedSliceGrad/begin?
4gradients/strided_slice_grad/MLCStridedSliceGrad/endConst*
_output_shapes
:*
dtype0*
valueB: 26
4gradients/strided_slice_grad/MLCStridedSliceGrad/end?
8gradients/strided_slice_grad/MLCStridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*
valueB:2:
8gradients/strided_slice_grad/MLCStridedSliceGrad/strides?
0gradients/strided_slice_grad/MLCStridedSliceGradMLCStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0?gradients/strided_slice_grad/MLCStridedSliceGrad/begin:output:0=gradients/strided_slice_grad/MLCStridedSliceGrad/end:output:0Agradients/strided_slice_grad/MLCStridedSliceGrad/strides:output:0gradients/grad_ys_0:output:0>gradients_strided_slice_grad_mlcstridedslicegrad_strided_slice*
Index0*
T0*,
_output_shapes
:??????????*
shrink_axis_mask22
0gradients/strided_slice_grad/MLCStridedSliceGrad?
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:2.
,gradients/transpose_7_grad/InvertPermutation?
+gradients/transpose_7_grad/MLCTransposeGradMLCTransposeGradgradients/grad_ys_1:output:00gradients/transpose_7_grad/InvertPermutation:y:07gradients_transpose_7_grad_mlctransposegrad_transpose_7*
T0*,
_output_shapes
:??????????2-
+gradients/transpose_7_grad/MLCTransposeGrad?
gradients/Squeeze_grad/ShapeShape%gradients_squeeze_grad_shape_cudnnrnn*
T0*
_output_shapes
:2
gradients/Squeeze_grad/Shape?
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*+
_output_shapes
:?????????2 
gradients/Squeeze_grad/Reshape?
gradients/AddNAddN9gradients/strided_slice_grad/MLCStridedSliceGrad:output:0/gradients/transpose_7_grad/MLCTransposeGrad:y:0*
N*
T0*C
_class9
75loc:@gradients/strided_slice_grad/MLCStridedSliceGrad*,
_output_shapes
:??????????2
gradients/AddNy
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
: 2
gradients/zeros_like?
gradients/zeros_like_1	ZerosLikegradients_zeros_like_1_cudnnrnn*
T0*
_output_shapes
:2
gradients/zeros_like_1?
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat+gradients_strided_slice_grad_shape_cudnnrnn%gradients_squeeze_grad_shape_cudnnrnngradients_zeros_like_cudnnrnngradients/AddN:sum:0'gradients/Squeeze_grad/Reshape:output:0gradients/zeros_like:y:0gradients_zeros_like_1_cudnnrnn*
T0*L
_output_shapes:
8:??????????:?????????: :?*
rnn_modegru2*
(gradients/CudnnRNN_grad/CudnnRNNBackprop?
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:2,
*gradients/transpose_grad/InvertPermutation?
)gradients/transpose_grad/MLCTransposeGradMLCTransposeGrad9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:02gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose*
T0*,
_output_shapes
:??????????2+
)gradients/transpose_grad/MLCTransposeGrad?
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
:2!
gradients/ExpandDims_grad/Shape?
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*'
_output_shapes
:?????????2#
!gradients/ExpandDims_grad/Reshape?
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:?2
gradients/concat_grad/Shape|
gradients/concat_grad/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_grad/add/y?
gradients/concat_grad/addAddV2%gradients_concat_grad_add_concat_axis$gradients/concat_grad/add/y:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/add?
)gradients/concat_grad/strided_slice/stackPack%gradients_concat_grad_add_concat_axis*
N*
T0*
_output_shapes
:2+
)gradients/concat_grad/strided_slice/stack?
+gradients/concat_grad/strided_slice/stack_1Packgradients/concat_grad/add:z:0*
N*
T0*
_output_shapes
:2-
+gradients/concat_grad/strided_slice/stack_1?
+gradients/concat_grad/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+gradients/concat_grad/strided_slice/stack_2?
#gradients/concat_grad/strided_sliceStridedSlice$gradients/concat_grad/Shape:output:02gradients/concat_grad/strided_slice/stack:output:04gradients/concat_grad/strided_slice/stack_1:output:04gradients/concat_grad/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#gradients/concat_grad/strided_slice?
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:?2
gradients/concat_grad/Shape_1?
gradients/concat_grad/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_grad/add_1/y?
gradients/concat_grad/add_1AddV2%gradients_concat_grad_add_concat_axis&gradients/concat_grad/add_1/y:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/add_1?
+gradients/concat_grad/strided_slice_1/stackPack%gradients_concat_grad_add_concat_axis*
N*
T0*
_output_shapes
:2-
+gradients/concat_grad/strided_slice_1/stack?
-gradients/concat_grad/strided_slice_1/stack_1Packgradients/concat_grad/add_1:z:0*
N*
T0*
_output_shapes
:2/
-gradients/concat_grad/strided_slice_1/stack_1?
-gradients/concat_grad/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-gradients/concat_grad/strided_slice_1/stack_2?
%gradients/concat_grad/strided_slice_1StridedSlice&gradients/concat_grad/Shape_1:output:04gradients/concat_grad/strided_slice_1/stack:output:06gradients/concat_grad/strided_slice_1/stack_1:output:06gradients/concat_grad/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%gradients/concat_grad/strided_slice_1?
gradients/concat_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:?2
gradients/concat_grad/Shape_2?
gradients/concat_grad/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_grad/add_2/y?
gradients/concat_grad/add_2AddV2%gradients_concat_grad_add_concat_axis&gradients/concat_grad/add_2/y:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/add_2?
+gradients/concat_grad/strided_slice_2/stackPack%gradients_concat_grad_add_concat_axis*
N*
T0*
_output_shapes
:2-
+gradients/concat_grad/strided_slice_2/stack?
-gradients/concat_grad/strided_slice_2/stack_1Packgradients/concat_grad/add_2:z:0*
N*
T0*
_output_shapes
:2/
-gradients/concat_grad/strided_slice_2/stack_1?
-gradients/concat_grad/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-gradients/concat_grad/strided_slice_2/stack_2?
%gradients/concat_grad/strided_slice_2StridedSlice&gradients/concat_grad/Shape_2:output:04gradients/concat_grad/strided_slice_2/stack:output:06gradients/concat_grad/strided_slice_2/stack_1:output:06gradients/concat_grad/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%gradients/concat_grad/strided_slice_2?
gradients/concat_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:?2
gradients/concat_grad/Shape_3?
gradients/concat_grad/add_3/yConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_grad/add_3/y?
gradients/concat_grad/add_3AddV2%gradients_concat_grad_add_concat_axis&gradients/concat_grad/add_3/y:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/add_3?
+gradients/concat_grad/strided_slice_3/stackPack%gradients_concat_grad_add_concat_axis*
N*
T0*
_output_shapes
:2-
+gradients/concat_grad/strided_slice_3/stack?
-gradients/concat_grad/strided_slice_3/stack_1Packgradients/concat_grad/add_3:z:0*
N*
T0*
_output_shapes
:2/
-gradients/concat_grad/strided_slice_3/stack_1?
-gradients/concat_grad/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-gradients/concat_grad/strided_slice_3/stack_2?
%gradients/concat_grad/strided_slice_3StridedSlice&gradients/concat_grad/Shape_3:output:04gradients/concat_grad/strided_slice_3/stack:output:06gradients/concat_grad/strided_slice_3/stack_1:output:06gradients/concat_grad/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%gradients/concat_grad/strided_slice_3?
gradients/concat_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:?2
gradients/concat_grad/Shape_4?
gradients/concat_grad/add_4/yConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_grad/add_4/y?
gradients/concat_grad/add_4AddV2%gradients_concat_grad_add_concat_axis&gradients/concat_grad/add_4/y:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/add_4?
+gradients/concat_grad/strided_slice_4/stackPack%gradients_concat_grad_add_concat_axis*
N*
T0*
_output_shapes
:2-
+gradients/concat_grad/strided_slice_4/stack?
-gradients/concat_grad/strided_slice_4/stack_1Packgradients/concat_grad/add_4:z:0*
N*
T0*
_output_shapes
:2/
-gradients/concat_grad/strided_slice_4/stack_1?
-gradients/concat_grad/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-gradients/concat_grad/strided_slice_4/stack_2?
%gradients/concat_grad/strided_slice_4StridedSlice&gradients/concat_grad/Shape_4:output:04gradients/concat_grad/strided_slice_4/stack:output:06gradients/concat_grad/strided_slice_4/stack_1:output:06gradients/concat_grad/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%gradients/concat_grad/strided_slice_4?
gradients/concat_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:?2
gradients/concat_grad/Shape_5?
gradients/concat_grad/add_5/yConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_grad/add_5/y?
gradients/concat_grad/add_5AddV2%gradients_concat_grad_add_concat_axis&gradients/concat_grad/add_5/y:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/add_5?
+gradients/concat_grad/strided_slice_5/stackPack%gradients_concat_grad_add_concat_axis*
N*
T0*
_output_shapes
:2-
+gradients/concat_grad/strided_slice_5/stack?
-gradients/concat_grad/strided_slice_5/stack_1Packgradients/concat_grad/add_5:z:0*
N*
T0*
_output_shapes
:2/
-gradients/concat_grad/strided_slice_5/stack_1?
-gradients/concat_grad/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-gradients/concat_grad/strided_slice_5/stack_2?
%gradients/concat_grad/strided_slice_5StridedSlice&gradients/concat_grad/Shape_5:output:04gradients/concat_grad/strided_slice_5/stack:output:06gradients/concat_grad/strided_slice_5/stack_1:output:06gradients/concat_grad/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%gradients/concat_grad/strided_slice_5?
gradients/concat_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape_6?
gradients/concat_grad/add_6/yConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_grad/add_6/y?
gradients/concat_grad/add_6AddV2%gradients_concat_grad_add_concat_axis&gradients/concat_grad/add_6/y:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/add_6?
+gradients/concat_grad/strided_slice_6/stackPack%gradients_concat_grad_add_concat_axis*
N*
T0*
_output_shapes
:2-
+gradients/concat_grad/strided_slice_6/stack?
-gradients/concat_grad/strided_slice_6/stack_1Packgradients/concat_grad/add_6:z:0*
N*
T0*
_output_shapes
:2/
-gradients/concat_grad/strided_slice_6/stack_1?
-gradients/concat_grad/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-gradients/concat_grad/strided_slice_6/stack_2?
%gradients/concat_grad/strided_slice_6StridedSlice&gradients/concat_grad/Shape_6:output:04gradients/concat_grad/strided_slice_6/stack:output:06gradients/concat_grad/strided_slice_6/stack_1:output:06gradients/concat_grad/strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%gradients/concat_grad/strided_slice_6?
gradients/concat_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape_7?
gradients/concat_grad/add_7/yConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_grad/add_7/y?
gradients/concat_grad/add_7AddV2%gradients_concat_grad_add_concat_axis&gradients/concat_grad/add_7/y:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/add_7?
+gradients/concat_grad/strided_slice_7/stackPack%gradients_concat_grad_add_concat_axis*
N*
T0*
_output_shapes
:2-
+gradients/concat_grad/strided_slice_7/stack?
-gradients/concat_grad/strided_slice_7/stack_1Packgradients/concat_grad/add_7:z:0*
N*
T0*
_output_shapes
:2/
-gradients/concat_grad/strided_slice_7/stack_1?
-gradients/concat_grad/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-gradients/concat_grad/strided_slice_7/stack_2?
%gradients/concat_grad/strided_slice_7StridedSlice&gradients/concat_grad/Shape_7:output:04gradients/concat_grad/strided_slice_7/stack:output:06gradients/concat_grad/strided_slice_7/stack_1:output:06gradients/concat_grad/strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%gradients/concat_grad/strided_slice_7?
gradients/concat_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape_8?
gradients/concat_grad/add_8/yConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_grad/add_8/y?
gradients/concat_grad/add_8AddV2%gradients_concat_grad_add_concat_axis&gradients/concat_grad/add_8/y:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/add_8?
+gradients/concat_grad/strided_slice_8/stackPack%gradients_concat_grad_add_concat_axis*
N*
T0*
_output_shapes
:2-
+gradients/concat_grad/strided_slice_8/stack?
-gradients/concat_grad/strided_slice_8/stack_1Packgradients/concat_grad/add_8:z:0*
N*
T0*
_output_shapes
:2/
-gradients/concat_grad/strided_slice_8/stack_1?
-gradients/concat_grad/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-gradients/concat_grad/strided_slice_8/stack_2?
%gradients/concat_grad/strided_slice_8StridedSlice&gradients/concat_grad/Shape_8:output:04gradients/concat_grad/strided_slice_8/stack:output:06gradients/concat_grad/strided_slice_8/stack_1:output:06gradients/concat_grad/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%gradients/concat_grad/strided_slice_8?
gradients/concat_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape_9?
gradients/concat_grad/add_9/yConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_grad/add_9/y?
gradients/concat_grad/add_9AddV2%gradients_concat_grad_add_concat_axis&gradients/concat_grad/add_9/y:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/add_9?
+gradients/concat_grad/strided_slice_9/stackPack%gradients_concat_grad_add_concat_axis*
N*
T0*
_output_shapes
:2-
+gradients/concat_grad/strided_slice_9/stack?
-gradients/concat_grad/strided_slice_9/stack_1Packgradients/concat_grad/add_9:z:0*
N*
T0*
_output_shapes
:2/
-gradients/concat_grad/strided_slice_9/stack_1?
-gradients/concat_grad/strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-gradients/concat_grad/strided_slice_9/stack_2?
%gradients/concat_grad/strided_slice_9StridedSlice&gradients/concat_grad/Shape_9:output:04gradients/concat_grad/strided_slice_9/stack:output:06gradients/concat_grad/strided_slice_9/stack_1:output:06gradients/concat_grad/strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%gradients/concat_grad/strided_slice_9?
gradients/concat_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB:2 
gradients/concat_grad/Shape_10?
gradients/concat_grad/add_10/yConst*
_output_shapes
: *
dtype0*
value	B :2 
gradients/concat_grad/add_10/y?
gradients/concat_grad/add_10AddV2%gradients_concat_grad_add_concat_axis'gradients/concat_grad/add_10/y:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/add_10?
,gradients/concat_grad/strided_slice_10/stackPack%gradients_concat_grad_add_concat_axis*
N*
T0*
_output_shapes
:2.
,gradients/concat_grad/strided_slice_10/stack?
.gradients/concat_grad/strided_slice_10/stack_1Pack gradients/concat_grad/add_10:z:0*
N*
T0*
_output_shapes
:20
.gradients/concat_grad/strided_slice_10/stack_1?
.gradients/concat_grad/strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.gradients/concat_grad/strided_slice_10/stack_2?
&gradients/concat_grad/strided_slice_10StridedSlice'gradients/concat_grad/Shape_10:output:05gradients/concat_grad/strided_slice_10/stack:output:07gradients/concat_grad/strided_slice_10/stack_1:output:07gradients/concat_grad/strided_slice_10/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2(
&gradients/concat_grad/strided_slice_10?
gradients/concat_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB:2 
gradients/concat_grad/Shape_11?
gradients/concat_grad/add_11/yConst*
_output_shapes
: *
dtype0*
value	B :2 
gradients/concat_grad/add_11/y?
gradients/concat_grad/add_11AddV2%gradients_concat_grad_add_concat_axis'gradients/concat_grad/add_11/y:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/add_11?
,gradients/concat_grad/strided_slice_11/stackPack%gradients_concat_grad_add_concat_axis*
N*
T0*
_output_shapes
:2.
,gradients/concat_grad/strided_slice_11/stack?
.gradients/concat_grad/strided_slice_11/stack_1Pack gradients/concat_grad/add_11:z:0*
N*
T0*
_output_shapes
:20
.gradients/concat_grad/strided_slice_11/stack_1?
.gradients/concat_grad/strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.gradients/concat_grad/strided_slice_11/stack_2?
&gradients/concat_grad/strided_slice_11StridedSlice'gradients/concat_grad/Shape_11:output:05gradients/concat_grad/strided_slice_11/stack:output:07gradients/concat_grad/strided_slice_11/stack_1:output:07gradients/concat_grad/strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2(
&gradients/concat_grad/strided_slice_11?
gradients/concat_grad/packedPack,gradients/concat_grad/strided_slice:output:0.gradients/concat_grad/strided_slice_1:output:0.gradients/concat_grad/strided_slice_2:output:0.gradients/concat_grad/strided_slice_3:output:0.gradients/concat_grad/strided_slice_4:output:0.gradients/concat_grad/strided_slice_5:output:0.gradients/concat_grad/strided_slice_6:output:0.gradients/concat_grad/strided_slice_7:output:0.gradients/concat_grad/strided_slice_8:output:0.gradients/concat_grad/strided_slice_9:output:0/gradients/concat_grad/strided_slice_10:output:0/gradients/concat_grad/strided_slice_11:output:0*
N*
T0*
_output_shapes
:2
gradients/concat_grad/packed?
%gradients/concat_grad/MLCConcatV2GradMLCConcatV2Grad:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0%gradients/concat_grad/packed:output:0%gradients_concat_grad_add_concat_axis$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0&gradients/concat_grad/Shape_2:output:0&gradients/concat_grad/Shape_3:output:0&gradients/concat_grad/Shape_4:output:0&gradients/concat_grad/Shape_5:output:0&gradients/concat_grad/Shape_6:output:0&gradients/concat_grad/Shape_7:output:0&gradients/concat_grad/Shape_8:output:0&gradients/concat_grad/Shape_9:output:0'gradients/concat_grad/Shape_10:output:0'gradients/concat_grad/Shape_11:output:0/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat*
N*
T0*

Tlen0*b
_output_shapesP
N:?:?:?:?:?:?::::::2'
%gradients/concat_grad/MLCConcatV2Grad?
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
gradients/Reshape_1_grad/Shape?
'gradients/Reshape_1_grad/MLCReshapeGradMLCReshapeGrad.gradients/concat_grad/MLCConcatV2Grad:output:0'gradients/Reshape_1_grad/Shape:output:01gradients_reshape_1_grad_mlcreshapegrad_reshape_1*
T0*
_output_shapes

:2)
'gradients/Reshape_1_grad/MLCReshapeGrad?
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
gradients/Reshape_2_grad/Shape?
'gradients/Reshape_2_grad/MLCReshapeGradMLCReshapeGrad.gradients/concat_grad/MLCConcatV2Grad:output:1'gradients/Reshape_2_grad/Shape:output:01gradients_reshape_2_grad_mlcreshapegrad_reshape_2*
T0*
_output_shapes

:2)
'gradients/Reshape_2_grad/MLCReshapeGrad?
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
gradients/Reshape_3_grad/Shape?
'gradients/Reshape_3_grad/MLCReshapeGradMLCReshapeGrad.gradients/concat_grad/MLCConcatV2Grad:output:2'gradients/Reshape_3_grad/Shape:output:01gradients_reshape_3_grad_mlcreshapegrad_reshape_3*
T0*
_output_shapes

:2)
'gradients/Reshape_3_grad/MLCReshapeGrad?
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
gradients/Reshape_4_grad/Shape?
'gradients/Reshape_4_grad/MLCReshapeGradMLCReshapeGrad.gradients/concat_grad/MLCConcatV2Grad:output:3'gradients/Reshape_4_grad/Shape:output:01gradients_reshape_4_grad_mlcreshapegrad_reshape_4*
T0*
_output_shapes

:2)
'gradients/Reshape_4_grad/MLCReshapeGrad?
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
gradients/Reshape_5_grad/Shape?
'gradients/Reshape_5_grad/MLCReshapeGradMLCReshapeGrad.gradients/concat_grad/MLCConcatV2Grad:output:4'gradients/Reshape_5_grad/Shape:output:01gradients_reshape_5_grad_mlcreshapegrad_reshape_5*
T0*
_output_shapes

:2)
'gradients/Reshape_5_grad/MLCReshapeGrad?
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
gradients/Reshape_6_grad/Shape?
'gradients/Reshape_6_grad/MLCReshapeGradMLCReshapeGrad.gradients/concat_grad/MLCConcatV2Grad:output:5'gradients/Reshape_6_grad/Shape:output:01gradients_reshape_6_grad_mlcreshapegrad_reshape_6*
T0*
_output_shapes

:2)
'gradients/Reshape_6_grad/MLCReshapeGrad?
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2 
gradients/Reshape_7_grad/Shape?
'gradients/Reshape_7_grad/MLCReshapeGradMLCReshapeGrad.gradients/concat_grad/MLCConcatV2Grad:output:6'gradients/Reshape_7_grad/Shape:output:01gradients_reshape_7_grad_mlcreshapegrad_reshape_7*
T0*
_output_shapes
:2)
'gradients/Reshape_7_grad/MLCReshapeGrad?
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2 
gradients/Reshape_8_grad/Shape?
'gradients/Reshape_8_grad/MLCReshapeGradMLCReshapeGrad.gradients/concat_grad/MLCConcatV2Grad:output:7'gradients/Reshape_8_grad/Shape:output:01gradients_reshape_8_grad_mlcreshapegrad_reshape_8*
T0*
_output_shapes
:2)
'gradients/Reshape_8_grad/MLCReshapeGrad?
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2 
gradients/Reshape_9_grad/Shape?
'gradients/Reshape_9_grad/MLCReshapeGradMLCReshapeGrad.gradients/concat_grad/MLCConcatV2Grad:output:8'gradients/Reshape_9_grad/Shape:output:01gradients_reshape_9_grad_mlcreshapegrad_reshape_9*
T0*
_output_shapes
:2)
'gradients/Reshape_9_grad/MLCReshapeGrad?
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2!
gradients/Reshape_10_grad/Shape?
(gradients/Reshape_10_grad/MLCReshapeGradMLCReshapeGrad.gradients/concat_grad/MLCConcatV2Grad:output:9(gradients/Reshape_10_grad/Shape:output:03gradients_reshape_10_grad_mlcreshapegrad_reshape_10*
T0*
_output_shapes
:2*
(gradients/Reshape_10_grad/MLCReshapeGrad?
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2!
gradients/Reshape_11_grad/Shape?
(gradients/Reshape_11_grad/MLCReshapeGradMLCReshapeGrad/gradients/concat_grad/MLCConcatV2Grad:output:10(gradients/Reshape_11_grad/Shape:output:03gradients_reshape_11_grad_mlcreshapegrad_reshape_11*
T0*
_output_shapes
:2*
(gradients/Reshape_11_grad/MLCReshapeGrad?
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2!
gradients/Reshape_12_grad/Shape?
(gradients/Reshape_12_grad/MLCReshapeGradMLCReshapeGrad/gradients/concat_grad/MLCConcatV2Grad:output:11(gradients/Reshape_12_grad/Shape:output:03gradients_reshape_12_grad_mlcreshapegrad_reshape_12*
T0*
_output_shapes
:2*
(gradients/Reshape_12_grad/MLCReshapeGrad?
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:2.
,gradients/transpose_1_grad/InvertPermutation?
+gradients/transpose_1_grad/MLCTransposeGradMLCTransposeGrad0gradients/Reshape_1_grad/MLCReshapeGrad:output:00gradients/transpose_1_grad/InvertPermutation:y:07gradients_transpose_1_grad_mlctransposegrad_transpose_1*
T0*
_output_shapes

:2-
+gradients/transpose_1_grad/MLCTransposeGrad?
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:2.
,gradients/transpose_2_grad/InvertPermutation?
+gradients/transpose_2_grad/MLCTransposeGradMLCTransposeGrad0gradients/Reshape_2_grad/MLCReshapeGrad:output:00gradients/transpose_2_grad/InvertPermutation:y:07gradients_transpose_2_grad_mlctransposegrad_transpose_2*
T0*
_output_shapes

:2-
+gradients/transpose_2_grad/MLCTransposeGrad?
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:2.
,gradients/transpose_3_grad/InvertPermutation?
+gradients/transpose_3_grad/MLCTransposeGradMLCTransposeGrad0gradients/Reshape_3_grad/MLCReshapeGrad:output:00gradients/transpose_3_grad/InvertPermutation:y:07gradients_transpose_3_grad_mlctransposegrad_transpose_3*
T0*
_output_shapes

:2-
+gradients/transpose_3_grad/MLCTransposeGrad?
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:2.
,gradients/transpose_4_grad/InvertPermutation?
+gradients/transpose_4_grad/MLCTransposeGradMLCTransposeGrad0gradients/Reshape_4_grad/MLCReshapeGrad:output:00gradients/transpose_4_grad/InvertPermutation:y:07gradients_transpose_4_grad_mlctransposegrad_transpose_4*
T0*
_output_shapes

:2-
+gradients/transpose_4_grad/MLCTransposeGrad?
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:2.
,gradients/transpose_5_grad/InvertPermutation?
+gradients/transpose_5_grad/MLCTransposeGradMLCTransposeGrad0gradients/Reshape_5_grad/MLCReshapeGrad:output:00gradients/transpose_5_grad/InvertPermutation:y:07gradients_transpose_5_grad_mlctransposegrad_transpose_5*
T0*
_output_shapes

:2-
+gradients/transpose_5_grad/MLCTransposeGrad?
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:2.
,gradients/transpose_6_grad/InvertPermutation?
+gradients/transpose_6_grad/MLCTransposeGradMLCTransposeGrad0gradients/Reshape_6_grad/MLCReshapeGrad:output:00gradients/transpose_6_grad/InvertPermutation:y:07gradients_transpose_6_grad_mlctransposegrad_transpose_6*
T0*
_output_shapes

:2-
+gradients/transpose_6_grad/MLCTransposeGrad?
gradients/split_2_grad/concatConcatV20gradients/Reshape_7_grad/MLCReshapeGrad:output:00gradients/Reshape_8_grad/MLCReshapeGrad:output:00gradients/Reshape_9_grad/MLCReshapeGrad:output:01gradients/Reshape_10_grad/MLCReshapeGrad:output:01gradients/Reshape_11_grad/MLCReshapeGrad:output:01gradients/Reshape_12_grad/MLCReshapeGrad:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes	
:?2
gradients/split_2_grad/concat?
!gradients/split_grad/MLCSplitGradMLCSplitGrad/gradients/transpose_1_grad/MLCTransposeGrad:y:0/gradients/transpose_2_grad/MLCTransposeGrad:y:0/gradients/transpose_3_grad/MLCTransposeGrad:y:01gradients_split_grad_mlcsplitgrad_split_split_dim'gradients_split_grad_mlcsplitgrad_split*
N*
T0*
_output_shapes

:Z2#
!gradients/split_grad/MLCSplitGrad?
#gradients/split_1_grad/MLCSplitGradMLCSplitGrad/gradients/transpose_4_grad/MLCTransposeGrad:y:0/gradients/transpose_5_grad/MLCTransposeGrad:y:0/gradients/transpose_6_grad/MLCTransposeGrad:y:05gradients_split_1_grad_mlcsplitgrad_split_1_split_dim+gradients_split_1_grad_mlcsplitgrad_split_1*
N*
T0*
_output_shapes

:Z2%
#gradients/split_1_grad/MLCSplitGrad?
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"   Z   2
gradients/Reshape_grad/Shape?
%gradients/Reshape_grad/MLCReshapeGradMLCReshapeGrad&gradients/split_2_grad/concat:output:0%gradients/Reshape_grad/Shape:output:0-gradients_reshape_grad_mlcreshapegrad_reshape*
T0*
_output_shapes

:Z2'
%gradients/Reshape_grad/MLCReshapeGrad?
IdentityIdentity-gradients/transpose_grad/MLCTransposeGrad:y:0*
T0*,
_output_shapes
:??????????2

Identity?

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0*
T0*'
_output_shapes
:?????????2

Identity_1y

Identity_2Identity*gradients/split_grad/MLCSplitGrad:output:0*
T0*
_output_shapes

:Z2

Identity_2{

Identity_3Identity,gradients/split_1_grad/MLCSplitGrad:output:0*
T0*
_output_shapes

:Z2

Identity_3}

Identity_4Identity.gradients/Reshape_grad/MLCReshapeGrad:output:0*
T0*
_output_shapes

:Z2

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*?
_input_shapes?
?:?????????:??????????:?????????: :??????????:?????????::??????????:?????????: ::??????????:?????????: :?::?????????: :?:?:?:?:?:?::::::::::::::::::: : :: ::?*<
api_implements*(gru_8d3b7182-ad7a-489b-86fa-c30b709435ce*
api_preferred_deviceGPU*@
forward_function_name'%__forward_gpu_gru_with_fallback_17132*
go_backwards( *

time_major( :- )
'
_output_shapes
:?????????:2.
,
_output_shapes
:??????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :2.
,
_output_shapes
:??????????:-)
'
_output_shapes
:?????????: 

_output_shapes
::2.
,
_output_shapes
:??????????:1-
+
_output_shapes
:?????????:	

_output_shapes
: :


_output_shapes
::2.
,
_output_shapes
:??????????:1-
+
_output_shapes
:?????????:

_output_shapes
: :!

_output_shapes	
:?: 

_output_shapes
::-)
'
_output_shapes
:?????????:

_output_shapes
: :!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::$ 

_output_shapes

::  

_output_shapes
::$! 

_output_shapes

:: "

_output_shapes
::$# 

_output_shapes

:: $

_output_shapes
::$% 

_output_shapes

:: &

_output_shapes
::$' 

_output_shapes

:: (

_output_shapes
::$) 

_output_shapes

::*

_output_shapes
: :+

_output_shapes
: :$, 

_output_shapes

::-

_output_shapes
: :$. 

_output_shapes

::!/

_output_shapes	
:?
?	
?
while_cond_17374
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice3
/while_while_cond_17374___redundant_placeholder03
/while_while_cond_17374___redundant_placeholder13
/while_while_cond_17374___redundant_placeholder23
/while_while_cond_17374___redundant_placeholder33
/while_while_cond_17374___redundant_placeholder4
while_identity
n

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*D
_input_shapes3
1: : : : :?????????: :::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
::

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
?<
?
'__inference_gpu_gru_with_fallback_16445

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3?u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm{
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:??????????2
	transposeb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims/dim}

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2

ExpandDimsP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
splitSplitsplit/split_dim:output:0kernel*
T0*2
_output_shapes 
:::*
	num_split2
splitT
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dim?
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*2
_output_shapes 
:::*
	num_split2	
split_1q
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
Reshape/shapea
ReshapeReshapebiasReshape/shape:output:0*
T0*
_output_shapes	
:?2	
ReshapeT
Const_2Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_2h
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_2/split_dim?
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*8
_output_shapes&
$::::::*
	num_split2	
split_2e
Const_3Const*
_output_shapes
:*
dtype0*
valueB:
?????????2	
Const_3u
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_1/perm{
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes

:2
transpose_1j
	Reshape_1Reshapetranspose_1:y:0Const_3:output:0*
T0*
_output_shapes	
:?2
	Reshape_1u
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_2/perm{
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes

:2
transpose_2j
	Reshape_2Reshapetranspose_2:y:0Const_3:output:0*
T0*
_output_shapes	
:?2
	Reshape_2u
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_3/perm{
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

:2
transpose_3j
	Reshape_3Reshapetranspose_3:y:0Const_3:output:0*
T0*
_output_shapes	
:?2
	Reshape_3u
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_4/perm}
transpose_4	Transposesplit_1:output:0transpose_4/perm:output:0*
T0*
_output_shapes

:2
transpose_4j
	Reshape_4Reshapetranspose_4:y:0Const_3:output:0*
T0*
_output_shapes	
:?2
	Reshape_4u
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_5/perm}
transpose_5	Transposesplit_1:output:1transpose_5/perm:output:0*
T0*
_output_shapes

:2
transpose_5j
	Reshape_5Reshapetranspose_5:y:0Const_3:output:0*
T0*
_output_shapes	
:?2
	Reshape_5u
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_6/perm}
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T0*
_output_shapes

:2
transpose_6j
	Reshape_6Reshapetranspose_6:y:0Const_3:output:0*
T0*
_output_shapes	
:?2
	Reshape_6j
	Reshape_7Reshapesplit_2:output:0Const_3:output:0*
T0*
_output_shapes
:2
	Reshape_7j
	Reshape_8Reshapesplit_2:output:1Const_3:output:0*
T0*
_output_shapes
:2
	Reshape_8j
	Reshape_9Reshapesplit_2:output:2Const_3:output:0*
T0*
_output_shapes
:2
	Reshape_9l

Reshape_10Reshapesplit_2:output:3Const_3:output:0*
T0*
_output_shapes
:2

Reshape_10l

Reshape_11Reshapesplit_2:output:4Const_3:output:0*
T0*
_output_shapes
:2

Reshape_11l

Reshape_12Reshapesplit_2:output:5Const_3:output:0*
T0*
_output_shapes
:2

Reshape_12\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat/axis?
concatConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
N*
T0*
_output_shapes	
:?2
concati
CudnnRNN/input_cConst*
_output_shapes
: *
dtype0*
valueB
 *    2
CudnnRNN/input_c?
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat:output:0*
T0*I
_output_shapes7
5:??????????:?????????: :*
rnn_modegru2

CudnnRNN}
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slicey
transpose_7/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_7/perm?
transpose_7	TransposeCudnnRNN:output:0transpose_7/perm:output:0*
T0*,
_output_shapes
:??????????2
transpose_7{
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:?????????*
squeeze_dims
 2	
Squeezef
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @2	
runtimej
IdentityIdentitystrided_slice:output:0*
T0*'
_output_shapes
:?????????2

Identityl

Identity_1Identitytranspose_7:y:0*
T0*,
_output_shapes
:??????????2

Identity_1h

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:?????????2

Identity_2W

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: 2

Identity_3"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*\
_input_shapesK
I:??????????:?????????:Z:Z:Z*<
api_implements*(gru_4820c462-f968-4106-b392-2a4aa0828265*
api_preferred_deviceGPU*
go_backwards( *

time_major( :T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinit_h:FB

_output_shapes

:Z
 
_user_specified_namekernel:PL

_output_shapes

:Z
*
_user_specified_namerecurrent_kernel:D@

_output_shapes

:Z

_user_specified_namebias
?
?
#__inference_gru_layer_call_fn_20199

inputs
unknown
	unknown_0
	unknown_1
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *G
fBR@
>__inference_gru_layer_call_and_return_conditional_losses_171352
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????:::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
G__inference_sequential_1_layer_call_and_return_conditional_losses_17193
	gru_input
	gru_17158
	gru_17160
	gru_17162
dense_2_17187
dense_2_17189
identity??dense_2/StatefulPartitionedCall?gru/StatefulPartitionedCall?
gru/StatefulPartitionedCallStatefulPartitionedCall	gru_input	gru_17158	gru_17160	gru_17162*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *G
fBR@
>__inference_gru_layer_call_and_return_conditional_losses_166662
gru/StatefulPartitionedCall?
dense_2/StatefulPartitionedCallStatefulPartitionedCall$gru/StatefulPartitionedCall:output:0dense_2_17187dense_2_17189*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_171762!
dense_2/StatefulPartitionedCall?
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0 ^dense_2/StatefulPartitionedCall^gru/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????:::::2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2:
gru/StatefulPartitionedCallgru/StatefulPartitionedCall:W S
,
_output_shapes
:??????????
#
_user_specified_name	gru_input
?M
?
%__forward_gpu_gru_with_fallback_19214

inputs
init_h_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3
cudnnrnn
strided_slice
transpose_7_perm
transpose_7

cudnnrnn_0

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
cudnnrnn_input_c

concat
transpose_perm

init_h
concat_axis
	reshape_1
	reshape_2
	reshape_3
	reshape_4
	reshape_5
	reshape_6
	reshape_7
	reshape_8
	reshape_9

reshape_10

reshape_11

reshape_12
transpose_1_perm
transpose_1
transpose_2_perm
transpose_2
transpose_3_perm
transpose_3
transpose_4_perm
transpose_4
transpose_5_perm
transpose_5
transpose_6_perm
transpose_6
split_2_split_dim
split_split_dim	
split
split_1_split_dim
split_1
reshape?u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permO
transpose_0	Transposeinputstranspose/perm:output:0*
T02
	transposeb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims/dim

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2

ExpandDimsP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dimU
split_0Splitsplit/split_dim:output:0kernel*
T0*
	num_split2
splitT
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dime
	split_1_0Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*
	num_split2	
split_1q
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
Reshape/shapea
ReshapeReshapebiasReshape/shape:output:0*
T0*
_output_shapes	
:?2	
ReshapeT
Const_2Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_2h
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_2/split_dim?
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*8
_output_shapes&
$::::::*
	num_split2	
split_2e
Const_3Const*
_output_shapes
:*
dtype0*
valueB:
?????????2	
Const_3u
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_1/perm_
transpose_1_0	Transposesplit_0:output:0transpose_1/perm:output:0*
T02
transpose_1l
	Reshape_1Reshapetranspose_1_0:y:0Const_3:output:0*
T0*
_output_shapes	
:?2
	Reshape_1u
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_2/perm_
transpose_2_0	Transposesplit_0:output:1transpose_2/perm:output:0*
T02
transpose_2l
	Reshape_2Reshapetranspose_2_0:y:0Const_3:output:0*
T0*
_output_shapes	
:?2
	Reshape_2u
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_3/perm_
transpose_3_0	Transposesplit_0:output:2transpose_3/perm:output:0*
T02
transpose_3l
	Reshape_3Reshapetranspose_3_0:y:0Const_3:output:0*
T0*
_output_shapes	
:?2
	Reshape_3u
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_4/perma
transpose_4_0	Transposesplit_1_0:output:0transpose_4/perm:output:0*
T02
transpose_4l
	Reshape_4Reshapetranspose_4_0:y:0Const_3:output:0*
T0*
_output_shapes	
:?2
	Reshape_4u
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_5/perma
transpose_5_0	Transposesplit_1_0:output:1transpose_5/perm:output:0*
T02
transpose_5l
	Reshape_5Reshapetranspose_5_0:y:0Const_3:output:0*
T0*
_output_shapes	
:?2
	Reshape_5u
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_6/perma
transpose_6_0	Transposesplit_1_0:output:2transpose_6/perm:output:0*
T02
transpose_6l
	Reshape_6Reshapetranspose_6_0:y:0Const_3:output:0*
T0*
_output_shapes	
:?2
	Reshape_6j
	Reshape_7Reshapesplit_2:output:0Const_3:output:0*
T0*
_output_shapes
:2
	Reshape_7j
	Reshape_8Reshapesplit_2:output:1Const_3:output:0*
T0*
_output_shapes
:2
	Reshape_8j
	Reshape_9Reshapesplit_2:output:2Const_3:output:0*
T0*
_output_shapes
:2
	Reshape_9l

Reshape_10Reshapesplit_2:output:3Const_3:output:0*
T0*
_output_shapes
:2

Reshape_10l

Reshape_11Reshapesplit_2:output:4Const_3:output:0*
T0*
_output_shapes
:2

Reshape_11l

Reshape_12Reshapesplit_2:output:5Const_3:output:0*
T0*
_output_shapes
:2

Reshape_12\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat/axis?
concat_0ConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
N*
T02
concati
CudnnRNN/input_cConst*
_output_shapes
: *
dtype0*
valueB
 *    2
CudnnRNN/input_c?
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat_0:output:0*
T0*Q
_output_shapes?
=:??????????????????:?????????: :*
rnn_modegru2

CudnnRNN}
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_slice_0StridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
shrink_axis_mask2
strided_slicey
transpose_7/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_7/perm`
transpose_7_0	TransposeCudnnRNN:output:0transpose_7/perm:output:0*
T02
transpose_7{
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:?????????*
squeeze_dims
 2	
Squeezef
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @2	
runtimel
IdentityIdentitystrided_slice_0:output:0*
T0*'
_output_shapes
:?????????2

Identityv

Identity_1Identitytranspose_7_0:y:0*
T0*4
_output_shapes"
 :??????????????????2

Identity_1h

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:?????????2

Identity_2W

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: 2

Identity_3"
concatconcat_0:output:0"#
concat_axisconcat/axis:output:0"
cudnnrnnCudnnRNN:output:0"!

cudnnrnn_0CudnnRNN:output_h:0"!

cudnnrnn_1CudnnRNN:output_c:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"-
cudnnrnn_input_cCudnnRNN/input_c:output:0"!

expanddimsExpandDims:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"
init_hinit_h_0"
reshapeReshape:output:0"
	reshape_1Reshape_1:output:0"!

reshape_10Reshape_10:output:0"!

reshape_11Reshape_11:output:0"!

reshape_12Reshape_12:output:0"
	reshape_2Reshape_2:output:0"
	reshape_3Reshape_3:output:0"
	reshape_4Reshape_4:output:0"
	reshape_5Reshape_5:output:0"
	reshape_6Reshape_6:output:0"
	reshape_7Reshape_7:output:0"
	reshape_8Reshape_8:output:0"
	reshape_9Reshape_9:output:0"
splitsplit_0:output:0"
split_1split_1_0:output:0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0")
strided_slicestrided_slice_0:output:0"
	transposetranspose_0:y:0" 
transpose_1transpose_1_0:y:0"-
transpose_1_permtranspose_1/perm:output:0" 
transpose_2transpose_2_0:y:0"-
transpose_2_permtranspose_2/perm:output:0" 
transpose_3transpose_3_0:y:0"-
transpose_3_permtranspose_3/perm:output:0" 
transpose_4transpose_4_0:y:0"-
transpose_4_permtranspose_4/perm:output:0" 
transpose_5transpose_5_0:y:0"-
transpose_5_permtranspose_5/perm:output:0" 
transpose_6transpose_6_0:y:0"-
transpose_6_permtranspose_6/perm:output:0" 
transpose_7transpose_7_0:y:0"-
transpose_7_permtranspose_7/perm:output:0")
transpose_permtranspose/perm:output:0*d
_input_shapesS
Q:??????????????????:?????????:Z:Z:Z*<
api_implements*(gru_d29f5daa-115b-49e1-809f-181a86e00a5d*
api_preferred_deviceGPU*T
backward_function_name:8__inference___backward_gpu_gru_with_fallback_18997_19215*
go_backwards( *

time_major( :\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinit_h:FB

_output_shapes

:Z
 
_user_specified_namekernel:PL

_output_shapes

:Z
*
_user_specified_namerecurrent_kernel:D@

_output_shapes

:Z

_user_specified_namebias
?2
?
while_body_19317
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_matmul_kernel_0
while_biasadd_unstack_0%
!while_matmul_1_recurrent_kernel_0
while_biasadd_1_unstack_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_matmul_kernel
while_biasadd_unstack#
while_matmul_1_recurrent_kernel
while_biasadd_1_unstack?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
while/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while_matmul_kernel_0*
T0*'
_output_shapes
:?????????Z2
while/MatMul?
while/BiasAddBiasAddwhile/MatMul:product:0while_biasadd_unstack_0*
T0*'
_output_shapes
:?????????Z2
while/BiasAdd\
while/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/Constp
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
while/split/split_dim?
while/splitSplitwhile/split/split_dim:output:0while/BiasAdd:output:0*
T0*M
_output_shapes;
9:?????????:?????????:?????????*
	num_split2
while/split?
while/MatMul_1MatMulwhile_placeholder_2!while_matmul_1_recurrent_kernel_0*
T0*'
_output_shapes
:?????????Z2
while/MatMul_1?
while/BiasAdd_1BiasAddwhile/MatMul_1:product:0while_biasadd_1_unstack_0*
T0*'
_output_shapes
:?????????Z2
while/BiasAdd_1`
while/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
while/Const_1t
while/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
while/split_1/split_dim?
while/split_1Split while/split_1/split_dim:output:0while/BiasAdd_1:output:0*
T0*M
_output_shapes;
9:?????????:?????????:?????????*
	num_split2
while/split_1
	while/addAddV2while/split:output:0while/split_1:output:0*
T0*'
_output_shapes
:?????????2
	while/addj
while/SigmoidSigmoidwhile/add:z:0*
T0*'
_output_shapes
:?????????2
while/Sigmoid?
while/add_1AddV2while/split:output:1while/split_1:output:1*
T0*'
_output_shapes
:?????????2
while/add_1p
while/Sigmoid_1Sigmoidwhile/add_1:z:0*
T0*'
_output_shapes
:?????????2
while/Sigmoid_1|
	while/mulMulwhile/Sigmoid_1:y:0while/split_1:output:2*
T0*'
_output_shapes
:?????????2
	while/mulz
while/add_2AddV2while/split:output:2while/mul:z:0*
T0*'
_output_shapes
:?????????2
while/add_2c

while/TanhTanhwhile/add_2:z:0*
T0*'
_output_shapes
:?????????2

while/Tanh{
while/mul_1Mulwhile/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:?????????2
while/mul_1_
while/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
while/sub/xx
	while/subSubwhile/sub/x:output:0while/Sigmoid:y:0*
T0*'
_output_shapes
:?????????2
	while/subr
while/mul_2Mulwhile/sub:z:0while/Tanh:y:0*
T0*'
_output_shapes
:?????????2
while/mul_2w
while/add_3AddV2while/mul_1:z:0while/mul_2:z:0*
T0*'
_output_shapes
:?????????2
while/add_3?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/add_3:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem`
while/add_4/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_4/yo
while/add_4AddV2while_placeholderwhile/add_4/y:output:0*
T0*
_output_shapes
: 2
while/add_4`
while/add_5/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_5/yv
while/add_5AddV2while_while_loop_counterwhile/add_5/y:output:0*
T0*
_output_shapes
: 2
while/add_5^
while/IdentityIdentitywhile/add_5:z:0*
T0*
_output_shapes
: 2
while/Identityq
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: 2
while/Identity_1b
while/Identity_2Identitywhile/add_4:z:0*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3s
while/Identity_4Identitywhile/add_3:z:0*
T0*'
_output_shapes
:?????????2
while/Identity_4"4
while_biasadd_1_unstackwhile_biasadd_1_unstack_0"0
while_biasadd_unstackwhile_biasadd_unstack_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"D
while_matmul_1_recurrent_kernel!while_matmul_1_recurrent_kernel_0",
while_matmul_kernelwhile_matmul_kernel_0",
while_strided_slicewhile_strided_slice_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*R
_input_shapesA
?: : : : :?????????: : :Z:Z:Z:Z: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:Z: 

_output_shapes
:Z:$	 

_output_shapes

:Z: 


_output_shapes
:Z
?
?
>__inference_gru_layer_call_and_return_conditional_losses_19708

inputs 
read_readvariableop_resource"
read_1_readvariableop_resource"
read_2_readvariableop_resource

identity_3??Read/ReadVariableOp?Read_1/ReadVariableOp?Read_2/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????2
zeros?
Read/ReadVariableOpReadVariableOpread_readvariableop_resource*
_output_shapes

:Z*
dtype02
Read/ReadVariableOpf
IdentityIdentityRead/ReadVariableOp:value:0*
T0*
_output_shapes

:Z2

Identity?
Read_1/ReadVariableOpReadVariableOpread_1_readvariableop_resource*
_output_shapes

:Z*
dtype02
Read_1/ReadVariableOpl

Identity_1IdentityRead_1/ReadVariableOp:value:0*
T0*
_output_shapes

:Z2

Identity_1?
Read_2/ReadVariableOpReadVariableOpread_2_readvariableop_resource*
_output_shapes

:Z*
dtype02
Read_2/ReadVariableOpl

Identity_2IdentityRead_2/ReadVariableOp:value:0*
T0*
_output_shapes

:Z2

Identity_2?
PartitionedCallPartitionedCallinputszeros:output:0Identity:output:0Identity_1:output:0Identity_2:output:0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *T
_output_shapesB
@:?????????:??????????:?????????: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *'
f"R 
__inference_standard_gru_194082
PartitionedCall?

Identity_3IdentityPartitionedCall:output:0^Read/ReadVariableOp^Read_1/ReadVariableOp^Read_2/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_3"!

identity_3Identity_3:output:0*7
_input_shapes&
$:??????????:::2*
Read/ReadVariableOpRead/ReadVariableOp2.
Read_1/ReadVariableOpRead_1/ReadVariableOp2.
Read_2/ReadVariableOpRead_2/ReadVariableOp:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
??
?
8__inference___backward_gpu_gru_with_fallback_19957_20175
placeholder
placeholder_1
placeholder_2
placeholder_3/
+gradients_strided_slice_grad_shape_cudnnrnnB
>gradients_strided_slice_grad_mlcstridedslicegrad_strided_sliceA
=gradients_transpose_7_grad_invertpermutation_transpose_7_perm;
7gradients_transpose_7_grad_mlctransposegrad_transpose_7)
%gradients_squeeze_grad_shape_cudnnrnn!
gradients_zeros_like_cudnnrnn#
gradients_zeros_like_1_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims=
9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c3
/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h)
%gradients_concat_grad_add_concat_axis5
1gradients_reshape_1_grad_mlcreshapegrad_reshape_15
1gradients_reshape_2_grad_mlcreshapegrad_reshape_25
1gradients_reshape_3_grad_mlcreshapegrad_reshape_35
1gradients_reshape_4_grad_mlcreshapegrad_reshape_45
1gradients_reshape_5_grad_mlcreshapegrad_reshape_55
1gradients_reshape_6_grad_mlcreshapegrad_reshape_65
1gradients_reshape_7_grad_mlcreshapegrad_reshape_75
1gradients_reshape_8_grad_mlcreshapegrad_reshape_85
1gradients_reshape_9_grad_mlcreshapegrad_reshape_97
3gradients_reshape_10_grad_mlcreshapegrad_reshape_107
3gradients_reshape_11_grad_mlcreshapegrad_reshape_117
3gradients_reshape_12_grad_mlcreshapegrad_reshape_12A
=gradients_transpose_1_grad_invertpermutation_transpose_1_perm;
7gradients_transpose_1_grad_mlctransposegrad_transpose_1A
=gradients_transpose_2_grad_invertpermutation_transpose_2_perm;
7gradients_transpose_2_grad_mlctransposegrad_transpose_2A
=gradients_transpose_3_grad_invertpermutation_transpose_3_perm;
7gradients_transpose_3_grad_mlctransposegrad_transpose_3A
=gradients_transpose_4_grad_invertpermutation_transpose_4_perm;
7gradients_transpose_4_grad_mlctransposegrad_transpose_4A
=gradients_transpose_5_grad_invertpermutation_transpose_5_perm;
7gradients_transpose_5_grad_mlctransposegrad_transpose_5A
=gradients_transpose_6_grad_invertpermutation_transpose_6_perm;
7gradients_transpose_6_grad_mlctransposegrad_transpose_63
/gradients_split_2_grad_concat_split_2_split_dim5
1gradients_split_grad_mlcsplitgrad_split_split_dim+
'gradients_split_grad_mlcsplitgrad_split9
5gradients_split_1_grad_mlcsplitgrad_split_1_split_dim/
+gradients_split_1_grad_mlcsplitgrad_split_11
-gradients_reshape_grad_mlcreshapegrad_reshape
identity

identity_1

identity_2

identity_3

identity_4?u
gradients/grad_ys_0Identityplaceholder*
T0*'
_output_shapes
:?????????2
gradients/grad_ys_0|
gradients/grad_ys_1Identityplaceholder_1*
T0*,
_output_shapes
:??????????2
gradients/grad_ys_1w
gradients/grad_ys_2Identityplaceholder_2*
T0*'
_output_shapes
:?????????2
gradients/grad_ys_2f
gradients/grad_ys_3Identityplaceholder_3*
T0*
_output_shapes
: 2
gradients/grad_ys_3?
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:2$
"gradients/strided_slice_grad/Shape?
6gradients/strided_slice_grad/MLCStridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*
valueB:
?????????28
6gradients/strided_slice_grad/MLCStridedSliceGrad/begin?
4gradients/strided_slice_grad/MLCStridedSliceGrad/endConst*
_output_shapes
:*
dtype0*
valueB: 26
4gradients/strided_slice_grad/MLCStridedSliceGrad/end?
8gradients/strided_slice_grad/MLCStridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*
valueB:2:
8gradients/strided_slice_grad/MLCStridedSliceGrad/strides?
0gradients/strided_slice_grad/MLCStridedSliceGradMLCStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0?gradients/strided_slice_grad/MLCStridedSliceGrad/begin:output:0=gradients/strided_slice_grad/MLCStridedSliceGrad/end:output:0Agradients/strided_slice_grad/MLCStridedSliceGrad/strides:output:0gradients/grad_ys_0:output:0>gradients_strided_slice_grad_mlcstridedslicegrad_strided_slice*
Index0*
T0*,
_output_shapes
:??????????*
shrink_axis_mask22
0gradients/strided_slice_grad/MLCStridedSliceGrad?
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:2.
,gradients/transpose_7_grad/InvertPermutation?
+gradients/transpose_7_grad/MLCTransposeGradMLCTransposeGradgradients/grad_ys_1:output:00gradients/transpose_7_grad/InvertPermutation:y:07gradients_transpose_7_grad_mlctransposegrad_transpose_7*
T0*,
_output_shapes
:??????????2-
+gradients/transpose_7_grad/MLCTransposeGrad?
gradients/Squeeze_grad/ShapeShape%gradients_squeeze_grad_shape_cudnnrnn*
T0*
_output_shapes
:2
gradients/Squeeze_grad/Shape?
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*+
_output_shapes
:?????????2 
gradients/Squeeze_grad/Reshape?
gradients/AddNAddN9gradients/strided_slice_grad/MLCStridedSliceGrad:output:0/gradients/transpose_7_grad/MLCTransposeGrad:y:0*
N*
T0*C
_class9
75loc:@gradients/strided_slice_grad/MLCStridedSliceGrad*,
_output_shapes
:??????????2
gradients/AddNy
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
: 2
gradients/zeros_like?
gradients/zeros_like_1	ZerosLikegradients_zeros_like_1_cudnnrnn*
T0*
_output_shapes
:2
gradients/zeros_like_1?
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat+gradients_strided_slice_grad_shape_cudnnrnn%gradients_squeeze_grad_shape_cudnnrnngradients_zeros_like_cudnnrnngradients/AddN:sum:0'gradients/Squeeze_grad/Reshape:output:0gradients/zeros_like:y:0gradients_zeros_like_1_cudnnrnn*
T0*L
_output_shapes:
8:??????????:?????????: :?*
rnn_modegru2*
(gradients/CudnnRNN_grad/CudnnRNNBackprop?
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:2,
*gradients/transpose_grad/InvertPermutation?
)gradients/transpose_grad/MLCTransposeGradMLCTransposeGrad9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:02gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose*
T0*,
_output_shapes
:??????????2+
)gradients/transpose_grad/MLCTransposeGrad?
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
:2!
gradients/ExpandDims_grad/Shape?
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*'
_output_shapes
:?????????2#
!gradients/ExpandDims_grad/Reshape?
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:?2
gradients/concat_grad/Shape|
gradients/concat_grad/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_grad/add/y?
gradients/concat_grad/addAddV2%gradients_concat_grad_add_concat_axis$gradients/concat_grad/add/y:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/add?
)gradients/concat_grad/strided_slice/stackPack%gradients_concat_grad_add_concat_axis*
N*
T0*
_output_shapes
:2+
)gradients/concat_grad/strided_slice/stack?
+gradients/concat_grad/strided_slice/stack_1Packgradients/concat_grad/add:z:0*
N*
T0*
_output_shapes
:2-
+gradients/concat_grad/strided_slice/stack_1?
+gradients/concat_grad/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+gradients/concat_grad/strided_slice/stack_2?
#gradients/concat_grad/strided_sliceStridedSlice$gradients/concat_grad/Shape:output:02gradients/concat_grad/strided_slice/stack:output:04gradients/concat_grad/strided_slice/stack_1:output:04gradients/concat_grad/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#gradients/concat_grad/strided_slice?
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:?2
gradients/concat_grad/Shape_1?
gradients/concat_grad/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_grad/add_1/y?
gradients/concat_grad/add_1AddV2%gradients_concat_grad_add_concat_axis&gradients/concat_grad/add_1/y:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/add_1?
+gradients/concat_grad/strided_slice_1/stackPack%gradients_concat_grad_add_concat_axis*
N*
T0*
_output_shapes
:2-
+gradients/concat_grad/strided_slice_1/stack?
-gradients/concat_grad/strided_slice_1/stack_1Packgradients/concat_grad/add_1:z:0*
N*
T0*
_output_shapes
:2/
-gradients/concat_grad/strided_slice_1/stack_1?
-gradients/concat_grad/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-gradients/concat_grad/strided_slice_1/stack_2?
%gradients/concat_grad/strided_slice_1StridedSlice&gradients/concat_grad/Shape_1:output:04gradients/concat_grad/strided_slice_1/stack:output:06gradients/concat_grad/strided_slice_1/stack_1:output:06gradients/concat_grad/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%gradients/concat_grad/strided_slice_1?
gradients/concat_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:?2
gradients/concat_grad/Shape_2?
gradients/concat_grad/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_grad/add_2/y?
gradients/concat_grad/add_2AddV2%gradients_concat_grad_add_concat_axis&gradients/concat_grad/add_2/y:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/add_2?
+gradients/concat_grad/strided_slice_2/stackPack%gradients_concat_grad_add_concat_axis*
N*
T0*
_output_shapes
:2-
+gradients/concat_grad/strided_slice_2/stack?
-gradients/concat_grad/strided_slice_2/stack_1Packgradients/concat_grad/add_2:z:0*
N*
T0*
_output_shapes
:2/
-gradients/concat_grad/strided_slice_2/stack_1?
-gradients/concat_grad/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-gradients/concat_grad/strided_slice_2/stack_2?
%gradients/concat_grad/strided_slice_2StridedSlice&gradients/concat_grad/Shape_2:output:04gradients/concat_grad/strided_slice_2/stack:output:06gradients/concat_grad/strided_slice_2/stack_1:output:06gradients/concat_grad/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%gradients/concat_grad/strided_slice_2?
gradients/concat_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:?2
gradients/concat_grad/Shape_3?
gradients/concat_grad/add_3/yConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_grad/add_3/y?
gradients/concat_grad/add_3AddV2%gradients_concat_grad_add_concat_axis&gradients/concat_grad/add_3/y:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/add_3?
+gradients/concat_grad/strided_slice_3/stackPack%gradients_concat_grad_add_concat_axis*
N*
T0*
_output_shapes
:2-
+gradients/concat_grad/strided_slice_3/stack?
-gradients/concat_grad/strided_slice_3/stack_1Packgradients/concat_grad/add_3:z:0*
N*
T0*
_output_shapes
:2/
-gradients/concat_grad/strided_slice_3/stack_1?
-gradients/concat_grad/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-gradients/concat_grad/strided_slice_3/stack_2?
%gradients/concat_grad/strided_slice_3StridedSlice&gradients/concat_grad/Shape_3:output:04gradients/concat_grad/strided_slice_3/stack:output:06gradients/concat_grad/strided_slice_3/stack_1:output:06gradients/concat_grad/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%gradients/concat_grad/strided_slice_3?
gradients/concat_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:?2
gradients/concat_grad/Shape_4?
gradients/concat_grad/add_4/yConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_grad/add_4/y?
gradients/concat_grad/add_4AddV2%gradients_concat_grad_add_concat_axis&gradients/concat_grad/add_4/y:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/add_4?
+gradients/concat_grad/strided_slice_4/stackPack%gradients_concat_grad_add_concat_axis*
N*
T0*
_output_shapes
:2-
+gradients/concat_grad/strided_slice_4/stack?
-gradients/concat_grad/strided_slice_4/stack_1Packgradients/concat_grad/add_4:z:0*
N*
T0*
_output_shapes
:2/
-gradients/concat_grad/strided_slice_4/stack_1?
-gradients/concat_grad/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-gradients/concat_grad/strided_slice_4/stack_2?
%gradients/concat_grad/strided_slice_4StridedSlice&gradients/concat_grad/Shape_4:output:04gradients/concat_grad/strided_slice_4/stack:output:06gradients/concat_grad/strided_slice_4/stack_1:output:06gradients/concat_grad/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%gradients/concat_grad/strided_slice_4?
gradients/concat_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:?2
gradients/concat_grad/Shape_5?
gradients/concat_grad/add_5/yConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_grad/add_5/y?
gradients/concat_grad/add_5AddV2%gradients_concat_grad_add_concat_axis&gradients/concat_grad/add_5/y:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/add_5?
+gradients/concat_grad/strided_slice_5/stackPack%gradients_concat_grad_add_concat_axis*
N*
T0*
_output_shapes
:2-
+gradients/concat_grad/strided_slice_5/stack?
-gradients/concat_grad/strided_slice_5/stack_1Packgradients/concat_grad/add_5:z:0*
N*
T0*
_output_shapes
:2/
-gradients/concat_grad/strided_slice_5/stack_1?
-gradients/concat_grad/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-gradients/concat_grad/strided_slice_5/stack_2?
%gradients/concat_grad/strided_slice_5StridedSlice&gradients/concat_grad/Shape_5:output:04gradients/concat_grad/strided_slice_5/stack:output:06gradients/concat_grad/strided_slice_5/stack_1:output:06gradients/concat_grad/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%gradients/concat_grad/strided_slice_5?
gradients/concat_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape_6?
gradients/concat_grad/add_6/yConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_grad/add_6/y?
gradients/concat_grad/add_6AddV2%gradients_concat_grad_add_concat_axis&gradients/concat_grad/add_6/y:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/add_6?
+gradients/concat_grad/strided_slice_6/stackPack%gradients_concat_grad_add_concat_axis*
N*
T0*
_output_shapes
:2-
+gradients/concat_grad/strided_slice_6/stack?
-gradients/concat_grad/strided_slice_6/stack_1Packgradients/concat_grad/add_6:z:0*
N*
T0*
_output_shapes
:2/
-gradients/concat_grad/strided_slice_6/stack_1?
-gradients/concat_grad/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-gradients/concat_grad/strided_slice_6/stack_2?
%gradients/concat_grad/strided_slice_6StridedSlice&gradients/concat_grad/Shape_6:output:04gradients/concat_grad/strided_slice_6/stack:output:06gradients/concat_grad/strided_slice_6/stack_1:output:06gradients/concat_grad/strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%gradients/concat_grad/strided_slice_6?
gradients/concat_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape_7?
gradients/concat_grad/add_7/yConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_grad/add_7/y?
gradients/concat_grad/add_7AddV2%gradients_concat_grad_add_concat_axis&gradients/concat_grad/add_7/y:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/add_7?
+gradients/concat_grad/strided_slice_7/stackPack%gradients_concat_grad_add_concat_axis*
N*
T0*
_output_shapes
:2-
+gradients/concat_grad/strided_slice_7/stack?
-gradients/concat_grad/strided_slice_7/stack_1Packgradients/concat_grad/add_7:z:0*
N*
T0*
_output_shapes
:2/
-gradients/concat_grad/strided_slice_7/stack_1?
-gradients/concat_grad/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-gradients/concat_grad/strided_slice_7/stack_2?
%gradients/concat_grad/strided_slice_7StridedSlice&gradients/concat_grad/Shape_7:output:04gradients/concat_grad/strided_slice_7/stack:output:06gradients/concat_grad/strided_slice_7/stack_1:output:06gradients/concat_grad/strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%gradients/concat_grad/strided_slice_7?
gradients/concat_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape_8?
gradients/concat_grad/add_8/yConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_grad/add_8/y?
gradients/concat_grad/add_8AddV2%gradients_concat_grad_add_concat_axis&gradients/concat_grad/add_8/y:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/add_8?
+gradients/concat_grad/strided_slice_8/stackPack%gradients_concat_grad_add_concat_axis*
N*
T0*
_output_shapes
:2-
+gradients/concat_grad/strided_slice_8/stack?
-gradients/concat_grad/strided_slice_8/stack_1Packgradients/concat_grad/add_8:z:0*
N*
T0*
_output_shapes
:2/
-gradients/concat_grad/strided_slice_8/stack_1?
-gradients/concat_grad/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-gradients/concat_grad/strided_slice_8/stack_2?
%gradients/concat_grad/strided_slice_8StridedSlice&gradients/concat_grad/Shape_8:output:04gradients/concat_grad/strided_slice_8/stack:output:06gradients/concat_grad/strided_slice_8/stack_1:output:06gradients/concat_grad/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%gradients/concat_grad/strided_slice_8?
gradients/concat_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape_9?
gradients/concat_grad/add_9/yConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_grad/add_9/y?
gradients/concat_grad/add_9AddV2%gradients_concat_grad_add_concat_axis&gradients/concat_grad/add_9/y:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/add_9?
+gradients/concat_grad/strided_slice_9/stackPack%gradients_concat_grad_add_concat_axis*
N*
T0*
_output_shapes
:2-
+gradients/concat_grad/strided_slice_9/stack?
-gradients/concat_grad/strided_slice_9/stack_1Packgradients/concat_grad/add_9:z:0*
N*
T0*
_output_shapes
:2/
-gradients/concat_grad/strided_slice_9/stack_1?
-gradients/concat_grad/strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-gradients/concat_grad/strided_slice_9/stack_2?
%gradients/concat_grad/strided_slice_9StridedSlice&gradients/concat_grad/Shape_9:output:04gradients/concat_grad/strided_slice_9/stack:output:06gradients/concat_grad/strided_slice_9/stack_1:output:06gradients/concat_grad/strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%gradients/concat_grad/strided_slice_9?
gradients/concat_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB:2 
gradients/concat_grad/Shape_10?
gradients/concat_grad/add_10/yConst*
_output_shapes
: *
dtype0*
value	B :2 
gradients/concat_grad/add_10/y?
gradients/concat_grad/add_10AddV2%gradients_concat_grad_add_concat_axis'gradients/concat_grad/add_10/y:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/add_10?
,gradients/concat_grad/strided_slice_10/stackPack%gradients_concat_grad_add_concat_axis*
N*
T0*
_output_shapes
:2.
,gradients/concat_grad/strided_slice_10/stack?
.gradients/concat_grad/strided_slice_10/stack_1Pack gradients/concat_grad/add_10:z:0*
N*
T0*
_output_shapes
:20
.gradients/concat_grad/strided_slice_10/stack_1?
.gradients/concat_grad/strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.gradients/concat_grad/strided_slice_10/stack_2?
&gradients/concat_grad/strided_slice_10StridedSlice'gradients/concat_grad/Shape_10:output:05gradients/concat_grad/strided_slice_10/stack:output:07gradients/concat_grad/strided_slice_10/stack_1:output:07gradients/concat_grad/strided_slice_10/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2(
&gradients/concat_grad/strided_slice_10?
gradients/concat_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB:2 
gradients/concat_grad/Shape_11?
gradients/concat_grad/add_11/yConst*
_output_shapes
: *
dtype0*
value	B :2 
gradients/concat_grad/add_11/y?
gradients/concat_grad/add_11AddV2%gradients_concat_grad_add_concat_axis'gradients/concat_grad/add_11/y:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/add_11?
,gradients/concat_grad/strided_slice_11/stackPack%gradients_concat_grad_add_concat_axis*
N*
T0*
_output_shapes
:2.
,gradients/concat_grad/strided_slice_11/stack?
.gradients/concat_grad/strided_slice_11/stack_1Pack gradients/concat_grad/add_11:z:0*
N*
T0*
_output_shapes
:20
.gradients/concat_grad/strided_slice_11/stack_1?
.gradients/concat_grad/strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.gradients/concat_grad/strided_slice_11/stack_2?
&gradients/concat_grad/strided_slice_11StridedSlice'gradients/concat_grad/Shape_11:output:05gradients/concat_grad/strided_slice_11/stack:output:07gradients/concat_grad/strided_slice_11/stack_1:output:07gradients/concat_grad/strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2(
&gradients/concat_grad/strided_slice_11?
gradients/concat_grad/packedPack,gradients/concat_grad/strided_slice:output:0.gradients/concat_grad/strided_slice_1:output:0.gradients/concat_grad/strided_slice_2:output:0.gradients/concat_grad/strided_slice_3:output:0.gradients/concat_grad/strided_slice_4:output:0.gradients/concat_grad/strided_slice_5:output:0.gradients/concat_grad/strided_slice_6:output:0.gradients/concat_grad/strided_slice_7:output:0.gradients/concat_grad/strided_slice_8:output:0.gradients/concat_grad/strided_slice_9:output:0/gradients/concat_grad/strided_slice_10:output:0/gradients/concat_grad/strided_slice_11:output:0*
N*
T0*
_output_shapes
:2
gradients/concat_grad/packed?
%gradients/concat_grad/MLCConcatV2GradMLCConcatV2Grad:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0%gradients/concat_grad/packed:output:0%gradients_concat_grad_add_concat_axis$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0&gradients/concat_grad/Shape_2:output:0&gradients/concat_grad/Shape_3:output:0&gradients/concat_grad/Shape_4:output:0&gradients/concat_grad/Shape_5:output:0&gradients/concat_grad/Shape_6:output:0&gradients/concat_grad/Shape_7:output:0&gradients/concat_grad/Shape_8:output:0&gradients/concat_grad/Shape_9:output:0'gradients/concat_grad/Shape_10:output:0'gradients/concat_grad/Shape_11:output:0/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat*
N*
T0*

Tlen0*b
_output_shapesP
N:?:?:?:?:?:?::::::2'
%gradients/concat_grad/MLCConcatV2Grad?
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
gradients/Reshape_1_grad/Shape?
'gradients/Reshape_1_grad/MLCReshapeGradMLCReshapeGrad.gradients/concat_grad/MLCConcatV2Grad:output:0'gradients/Reshape_1_grad/Shape:output:01gradients_reshape_1_grad_mlcreshapegrad_reshape_1*
T0*
_output_shapes

:2)
'gradients/Reshape_1_grad/MLCReshapeGrad?
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
gradients/Reshape_2_grad/Shape?
'gradients/Reshape_2_grad/MLCReshapeGradMLCReshapeGrad.gradients/concat_grad/MLCConcatV2Grad:output:1'gradients/Reshape_2_grad/Shape:output:01gradients_reshape_2_grad_mlcreshapegrad_reshape_2*
T0*
_output_shapes

:2)
'gradients/Reshape_2_grad/MLCReshapeGrad?
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
gradients/Reshape_3_grad/Shape?
'gradients/Reshape_3_grad/MLCReshapeGradMLCReshapeGrad.gradients/concat_grad/MLCConcatV2Grad:output:2'gradients/Reshape_3_grad/Shape:output:01gradients_reshape_3_grad_mlcreshapegrad_reshape_3*
T0*
_output_shapes

:2)
'gradients/Reshape_3_grad/MLCReshapeGrad?
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
gradients/Reshape_4_grad/Shape?
'gradients/Reshape_4_grad/MLCReshapeGradMLCReshapeGrad.gradients/concat_grad/MLCConcatV2Grad:output:3'gradients/Reshape_4_grad/Shape:output:01gradients_reshape_4_grad_mlcreshapegrad_reshape_4*
T0*
_output_shapes

:2)
'gradients/Reshape_4_grad/MLCReshapeGrad?
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
gradients/Reshape_5_grad/Shape?
'gradients/Reshape_5_grad/MLCReshapeGradMLCReshapeGrad.gradients/concat_grad/MLCConcatV2Grad:output:4'gradients/Reshape_5_grad/Shape:output:01gradients_reshape_5_grad_mlcreshapegrad_reshape_5*
T0*
_output_shapes

:2)
'gradients/Reshape_5_grad/MLCReshapeGrad?
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
gradients/Reshape_6_grad/Shape?
'gradients/Reshape_6_grad/MLCReshapeGradMLCReshapeGrad.gradients/concat_grad/MLCConcatV2Grad:output:5'gradients/Reshape_6_grad/Shape:output:01gradients_reshape_6_grad_mlcreshapegrad_reshape_6*
T0*
_output_shapes

:2)
'gradients/Reshape_6_grad/MLCReshapeGrad?
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2 
gradients/Reshape_7_grad/Shape?
'gradients/Reshape_7_grad/MLCReshapeGradMLCReshapeGrad.gradients/concat_grad/MLCConcatV2Grad:output:6'gradients/Reshape_7_grad/Shape:output:01gradients_reshape_7_grad_mlcreshapegrad_reshape_7*
T0*
_output_shapes
:2)
'gradients/Reshape_7_grad/MLCReshapeGrad?
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2 
gradients/Reshape_8_grad/Shape?
'gradients/Reshape_8_grad/MLCReshapeGradMLCReshapeGrad.gradients/concat_grad/MLCConcatV2Grad:output:7'gradients/Reshape_8_grad/Shape:output:01gradients_reshape_8_grad_mlcreshapegrad_reshape_8*
T0*
_output_shapes
:2)
'gradients/Reshape_8_grad/MLCReshapeGrad?
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2 
gradients/Reshape_9_grad/Shape?
'gradients/Reshape_9_grad/MLCReshapeGradMLCReshapeGrad.gradients/concat_grad/MLCConcatV2Grad:output:8'gradients/Reshape_9_grad/Shape:output:01gradients_reshape_9_grad_mlcreshapegrad_reshape_9*
T0*
_output_shapes
:2)
'gradients/Reshape_9_grad/MLCReshapeGrad?
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2!
gradients/Reshape_10_grad/Shape?
(gradients/Reshape_10_grad/MLCReshapeGradMLCReshapeGrad.gradients/concat_grad/MLCConcatV2Grad:output:9(gradients/Reshape_10_grad/Shape:output:03gradients_reshape_10_grad_mlcreshapegrad_reshape_10*
T0*
_output_shapes
:2*
(gradients/Reshape_10_grad/MLCReshapeGrad?
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2!
gradients/Reshape_11_grad/Shape?
(gradients/Reshape_11_grad/MLCReshapeGradMLCReshapeGrad/gradients/concat_grad/MLCConcatV2Grad:output:10(gradients/Reshape_11_grad/Shape:output:03gradients_reshape_11_grad_mlcreshapegrad_reshape_11*
T0*
_output_shapes
:2*
(gradients/Reshape_11_grad/MLCReshapeGrad?
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2!
gradients/Reshape_12_grad/Shape?
(gradients/Reshape_12_grad/MLCReshapeGradMLCReshapeGrad/gradients/concat_grad/MLCConcatV2Grad:output:11(gradients/Reshape_12_grad/Shape:output:03gradients_reshape_12_grad_mlcreshapegrad_reshape_12*
T0*
_output_shapes
:2*
(gradients/Reshape_12_grad/MLCReshapeGrad?
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:2.
,gradients/transpose_1_grad/InvertPermutation?
+gradients/transpose_1_grad/MLCTransposeGradMLCTransposeGrad0gradients/Reshape_1_grad/MLCReshapeGrad:output:00gradients/transpose_1_grad/InvertPermutation:y:07gradients_transpose_1_grad_mlctransposegrad_transpose_1*
T0*
_output_shapes

:2-
+gradients/transpose_1_grad/MLCTransposeGrad?
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:2.
,gradients/transpose_2_grad/InvertPermutation?
+gradients/transpose_2_grad/MLCTransposeGradMLCTransposeGrad0gradients/Reshape_2_grad/MLCReshapeGrad:output:00gradients/transpose_2_grad/InvertPermutation:y:07gradients_transpose_2_grad_mlctransposegrad_transpose_2*
T0*
_output_shapes

:2-
+gradients/transpose_2_grad/MLCTransposeGrad?
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:2.
,gradients/transpose_3_grad/InvertPermutation?
+gradients/transpose_3_grad/MLCTransposeGradMLCTransposeGrad0gradients/Reshape_3_grad/MLCReshapeGrad:output:00gradients/transpose_3_grad/InvertPermutation:y:07gradients_transpose_3_grad_mlctransposegrad_transpose_3*
T0*
_output_shapes

:2-
+gradients/transpose_3_grad/MLCTransposeGrad?
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:2.
,gradients/transpose_4_grad/InvertPermutation?
+gradients/transpose_4_grad/MLCTransposeGradMLCTransposeGrad0gradients/Reshape_4_grad/MLCReshapeGrad:output:00gradients/transpose_4_grad/InvertPermutation:y:07gradients_transpose_4_grad_mlctransposegrad_transpose_4*
T0*
_output_shapes

:2-
+gradients/transpose_4_grad/MLCTransposeGrad?
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:2.
,gradients/transpose_5_grad/InvertPermutation?
+gradients/transpose_5_grad/MLCTransposeGradMLCTransposeGrad0gradients/Reshape_5_grad/MLCReshapeGrad:output:00gradients/transpose_5_grad/InvertPermutation:y:07gradients_transpose_5_grad_mlctransposegrad_transpose_5*
T0*
_output_shapes

:2-
+gradients/transpose_5_grad/MLCTransposeGrad?
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:2.
,gradients/transpose_6_grad/InvertPermutation?
+gradients/transpose_6_grad/MLCTransposeGradMLCTransposeGrad0gradients/Reshape_6_grad/MLCReshapeGrad:output:00gradients/transpose_6_grad/InvertPermutation:y:07gradients_transpose_6_grad_mlctransposegrad_transpose_6*
T0*
_output_shapes

:2-
+gradients/transpose_6_grad/MLCTransposeGrad?
gradients/split_2_grad/concatConcatV20gradients/Reshape_7_grad/MLCReshapeGrad:output:00gradients/Reshape_8_grad/MLCReshapeGrad:output:00gradients/Reshape_9_grad/MLCReshapeGrad:output:01gradients/Reshape_10_grad/MLCReshapeGrad:output:01gradients/Reshape_11_grad/MLCReshapeGrad:output:01gradients/Reshape_12_grad/MLCReshapeGrad:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes	
:?2
gradients/split_2_grad/concat?
!gradients/split_grad/MLCSplitGradMLCSplitGrad/gradients/transpose_1_grad/MLCTransposeGrad:y:0/gradients/transpose_2_grad/MLCTransposeGrad:y:0/gradients/transpose_3_grad/MLCTransposeGrad:y:01gradients_split_grad_mlcsplitgrad_split_split_dim'gradients_split_grad_mlcsplitgrad_split*
N*
T0*
_output_shapes

:Z2#
!gradients/split_grad/MLCSplitGrad?
#gradients/split_1_grad/MLCSplitGradMLCSplitGrad/gradients/transpose_4_grad/MLCTransposeGrad:y:0/gradients/transpose_5_grad/MLCTransposeGrad:y:0/gradients/transpose_6_grad/MLCTransposeGrad:y:05gradients_split_1_grad_mlcsplitgrad_split_1_split_dim+gradients_split_1_grad_mlcsplitgrad_split_1*
N*
T0*
_output_shapes

:Z2%
#gradients/split_1_grad/MLCSplitGrad?
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"   Z   2
gradients/Reshape_grad/Shape?
%gradients/Reshape_grad/MLCReshapeGradMLCReshapeGrad&gradients/split_2_grad/concat:output:0%gradients/Reshape_grad/Shape:output:0-gradients_reshape_grad_mlcreshapegrad_reshape*
T0*
_output_shapes

:Z2'
%gradients/Reshape_grad/MLCReshapeGrad?
IdentityIdentity-gradients/transpose_grad/MLCTransposeGrad:y:0*
T0*,
_output_shapes
:??????????2

Identity?

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0*
T0*'
_output_shapes
:?????????2

Identity_1y

Identity_2Identity*gradients/split_grad/MLCSplitGrad:output:0*
T0*
_output_shapes

:Z2

Identity_2{

Identity_3Identity,gradients/split_1_grad/MLCSplitGrad:output:0*
T0*
_output_shapes

:Z2

Identity_3}

Identity_4Identity.gradients/Reshape_grad/MLCReshapeGrad:output:0*
T0*
_output_shapes

:Z2

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*?
_input_shapes?
?:?????????:??????????:?????????: :??????????:?????????::??????????:?????????: ::??????????:?????????: :?::?????????: :?:?:?:?:?:?::::::::::::::::::: : :: ::?*<
api_implements*(gru_c4a462d1-fe50-45b7-baf6-20215ecbec52*
api_preferred_deviceGPU*@
forward_function_name'%__forward_gpu_gru_with_fallback_20174*
go_backwards( *

time_major( :- )
'
_output_shapes
:?????????:2.
,
_output_shapes
:??????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :2.
,
_output_shapes
:??????????:-)
'
_output_shapes
:?????????: 

_output_shapes
::2.
,
_output_shapes
:??????????:1-
+
_output_shapes
:?????????:	

_output_shapes
: :


_output_shapes
::2.
,
_output_shapes
:??????????:1-
+
_output_shapes
:?????????:

_output_shapes
: :!

_output_shapes	
:?: 

_output_shapes
::-)
'
_output_shapes
:?????????:

_output_shapes
: :!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::$ 

_output_shapes

::  

_output_shapes
::$! 

_output_shapes

:: "

_output_shapes
::$# 

_output_shapes

:: $

_output_shapes
::$% 

_output_shapes

:: &

_output_shapes
::$' 

_output_shapes

:: (

_output_shapes
::$) 

_output_shapes

::*

_output_shapes
: :+

_output_shapes
: :$, 

_output_shapes

::-

_output_shapes
: :$. 

_output_shapes

::!/

_output_shapes	
:?
?n
?
!__inference__traced_restore_20408
file_prefix#
assignvariableop_dense_2_kernel#
assignvariableop_1_dense_2_bias 
assignvariableop_2_adam_iter"
assignvariableop_3_adam_beta_1"
assignvariableop_4_adam_beta_2!
assignvariableop_5_adam_decay)
%assignvariableop_6_adam_learning_rate*
&assignvariableop_7_gru_gru_cell_kernel4
0assignvariableop_8_gru_gru_cell_recurrent_kernel(
$assignvariableop_9_gru_gru_cell_bias
assignvariableop_10_total
assignvariableop_11_count
assignvariableop_12_total_1
assignvariableop_13_count_1
assignvariableop_14_total_2
assignvariableop_15_count_2-
)assignvariableop_16_adam_dense_2_kernel_m+
'assignvariableop_17_adam_dense_2_bias_m2
.assignvariableop_18_adam_gru_gru_cell_kernel_m<
8assignvariableop_19_adam_gru_gru_cell_recurrent_kernel_m0
,assignvariableop_20_adam_gru_gru_cell_bias_m-
)assignvariableop_21_adam_dense_2_kernel_v+
'assignvariableop_22_adam_dense_2_bias_v2
.assignvariableop_23_adam_gru_gru_cell_kernel_v<
8assignvariableop_24_adam_gru_gru_cell_recurrent_kernel_v0
,assignvariableop_25_adam_gru_gru_cell_bias_v
identity_27??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*I
value@B>B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapesn
l:::::::::::::::::::::::::::*)
dtypes
2	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOpassignvariableop_dense_2_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_2_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOpassignvariableop_2_adam_iterIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOpassignvariableop_3_adam_beta_1Identity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_beta_2Identity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_decayIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp%assignvariableop_6_adam_learning_rateIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp&assignvariableop_7_gru_gru_cell_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp0assignvariableop_8_gru_gru_cell_recurrent_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp$assignvariableop_9_gru_gru_cell_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOpassignvariableop_10_totalIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOpassignvariableop_11_countIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOpassignvariableop_12_total_1Identity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOpassignvariableop_13_count_1Identity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOpassignvariableop_14_total_2Identity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOpassignvariableop_15_count_2Identity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp)assignvariableop_16_adam_dense_2_kernel_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp'assignvariableop_17_adam_dense_2_bias_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp.assignvariableop_18_adam_gru_gru_cell_kernel_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp8assignvariableop_19_adam_gru_gru_cell_recurrent_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp,assignvariableop_20_adam_gru_gru_cell_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp)assignvariableop_21_adam_dense_2_kernel_vIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp'assignvariableop_22_adam_dense_2_bias_vIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp.assignvariableop_23_adam_gru_gru_cell_kernel_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp8assignvariableop_24_adam_gru_gru_cell_recurrent_kernel_vIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp,assignvariableop_25_adam_gru_gru_cell_bias_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_259
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_26Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_26?
Identity_27IdentityIdentity_26:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_27"#
identity_27Identity_27:output:0*}
_input_shapesl
j: ::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?E
?
__inference_standard_gru_19408

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3X
unstackUnpackbias*
T0* 
_output_shapes
:Z:Z*	
num2	
unstacku
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm{
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:??????????2
	transposeK
ShapeShapetranspose:y:0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_1n
MatMulMatMulstrided_slice_1:output:0kernel*
T0*'
_output_shapes
:?????????Z2
MatMuls
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*'
_output_shapes
:?????????Z2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:?????????:?????????:?????????*
	num_split2
splitj
MatMul_1MatMulinit_hrecurrent_kernel*
T0*'
_output_shapes
:?????????Z2

MatMul_1y
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*'
_output_shapes
:?????????Z2
	BiasAdd_1T
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dim?
split_1Splitsplit_1/split_dim:output:0BiasAdd_1:output:0*
T0*M
_output_shapes;
9:?????????:?????????:?????????*
	num_split2	
split_1g
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:?????????2
addX
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:?????????2	
Sigmoidk
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:?????????2
add_1^
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:?????????2
	Sigmoid_1d
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:?????????2
mulb
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:?????????2
add_2Q
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:?????????2
Tanh\
mul_1MulSigmoid:y:0init_h*
T0*'
_output_shapes
:?????????2
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
sub/x`
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:?????????2
subZ
mul_2Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:?????????2
mul_2_
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:?????????2
add_3?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileStatelessWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelunstack:output:0recurrent_kernelunstack:output:1*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*S
_output_shapesA
?: : : : :?????????: : :Z:Z:Z:Z* 
_read_only_resource_inputs
 *
bodyR
while_body_19317*
condR
while_cond_19316*R
output_shapesA
?: : : : :?????????: : :Z:Z:Z:Z*
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_2y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:??????????2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  ??2	
runtimel
IdentityIdentitystrided_slice_2:output:0*
T0*'
_output_shapes
:?????????2

Identityl

Identity_1Identitytranspose_1:y:0*
T0*,
_output_shapes
:??????????2

Identity_1f

Identity_2Identitywhile:output:4*
T0*'
_output_shapes
:?????????2

Identity_2W

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: 2

Identity_3"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*\
_input_shapesK
I:??????????:?????????:Z:Z:Z*<
api_implements*(gru_e9d87f5a-6bce-4ea5-b775-fb635bd39d20*
api_preferred_deviceCPU*
go_backwards( *

time_major( :T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinit_h:FB

_output_shapes

:Z
 
_user_specified_namekernel:PL

_output_shapes

:Z
*
_user_specified_namerecurrent_kernel:D@

_output_shapes

:Z

_user_specified_namebias
?2
?
while_body_17375
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_matmul_kernel_0
while_biasadd_unstack_0%
!while_matmul_1_recurrent_kernel_0
while_biasadd_1_unstack_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_matmul_kernel
while_biasadd_unstack#
while_matmul_1_recurrent_kernel
while_biasadd_1_unstack?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
while/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while_matmul_kernel_0*
T0*'
_output_shapes
:?????????Z2
while/MatMul?
while/BiasAddBiasAddwhile/MatMul:product:0while_biasadd_unstack_0*
T0*'
_output_shapes
:?????????Z2
while/BiasAdd\
while/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/Constp
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
while/split/split_dim?
while/splitSplitwhile/split/split_dim:output:0while/BiasAdd:output:0*
T0*M
_output_shapes;
9:?????????:?????????:?????????*
	num_split2
while/split?
while/MatMul_1MatMulwhile_placeholder_2!while_matmul_1_recurrent_kernel_0*
T0*'
_output_shapes
:?????????Z2
while/MatMul_1?
while/BiasAdd_1BiasAddwhile/MatMul_1:product:0while_biasadd_1_unstack_0*
T0*'
_output_shapes
:?????????Z2
while/BiasAdd_1`
while/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
while/Const_1t
while/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
while/split_1/split_dim?
while/split_1Split while/split_1/split_dim:output:0while/BiasAdd_1:output:0*
T0*M
_output_shapes;
9:?????????:?????????:?????????*
	num_split2
while/split_1
	while/addAddV2while/split:output:0while/split_1:output:0*
T0*'
_output_shapes
:?????????2
	while/addj
while/SigmoidSigmoidwhile/add:z:0*
T0*'
_output_shapes
:?????????2
while/Sigmoid?
while/add_1AddV2while/split:output:1while/split_1:output:1*
T0*'
_output_shapes
:?????????2
while/add_1p
while/Sigmoid_1Sigmoidwhile/add_1:z:0*
T0*'
_output_shapes
:?????????2
while/Sigmoid_1|
	while/mulMulwhile/Sigmoid_1:y:0while/split_1:output:2*
T0*'
_output_shapes
:?????????2
	while/mulz
while/add_2AddV2while/split:output:2while/mul:z:0*
T0*'
_output_shapes
:?????????2
while/add_2c

while/TanhTanhwhile/add_2:z:0*
T0*'
_output_shapes
:?????????2

while/Tanh{
while/mul_1Mulwhile/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:?????????2
while/mul_1_
while/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
while/sub/xx
	while/subSubwhile/sub/x:output:0while/Sigmoid:y:0*
T0*'
_output_shapes
:?????????2
	while/subr
while/mul_2Mulwhile/sub:z:0while/Tanh:y:0*
T0*'
_output_shapes
:?????????2
while/mul_2w
while/add_3AddV2while/mul_1:z:0while/mul_2:z:0*
T0*'
_output_shapes
:?????????2
while/add_3?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/add_3:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem`
while/add_4/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_4/yo
while/add_4AddV2while_placeholderwhile/add_4/y:output:0*
T0*
_output_shapes
: 2
while/add_4`
while/add_5/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_5/yv
while/add_5AddV2while_while_loop_counterwhile/add_5/y:output:0*
T0*
_output_shapes
: 2
while/add_5^
while/IdentityIdentitywhile/add_5:z:0*
T0*
_output_shapes
: 2
while/Identityq
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: 2
while/Identity_1b
while/Identity_2Identitywhile/add_4:z:0*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3s
while/Identity_4Identitywhile/add_3:z:0*
T0*'
_output_shapes
:?????????2
while/Identity_4"4
while_biasadd_1_unstackwhile_biasadd_1_unstack_0"0
while_biasadd_unstackwhile_biasadd_unstack_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"D
while_matmul_1_recurrent_kernel!while_matmul_1_recurrent_kernel_0",
while_matmul_kernelwhile_matmul_kernel_0",
while_strided_slicewhile_strided_slice_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*R
_input_shapesA
?: : : : :?????????: : :Z:Z:Z:Z: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:Z: 

_output_shapes
:Z:$	 

_output_shapes

:Z: 


_output_shapes
:Z
??
?
8__inference___backward_gpu_gru_with_fallback_14067_14285
placeholder
placeholder_1
placeholder_2
placeholder_3/
+gradients_strided_slice_grad_shape_cudnnrnnB
>gradients_strided_slice_grad_mlcstridedslicegrad_strided_sliceA
=gradients_transpose_7_grad_invertpermutation_transpose_7_perm;
7gradients_transpose_7_grad_mlctransposegrad_transpose_7)
%gradients_squeeze_grad_shape_cudnnrnn!
gradients_zeros_like_cudnnrnn#
gradients_zeros_like_1_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims=
9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c3
/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h)
%gradients_concat_grad_add_concat_axis5
1gradients_reshape_1_grad_mlcreshapegrad_reshape_15
1gradients_reshape_2_grad_mlcreshapegrad_reshape_25
1gradients_reshape_3_grad_mlcreshapegrad_reshape_35
1gradients_reshape_4_grad_mlcreshapegrad_reshape_45
1gradients_reshape_5_grad_mlcreshapegrad_reshape_55
1gradients_reshape_6_grad_mlcreshapegrad_reshape_65
1gradients_reshape_7_grad_mlcreshapegrad_reshape_75
1gradients_reshape_8_grad_mlcreshapegrad_reshape_85
1gradients_reshape_9_grad_mlcreshapegrad_reshape_97
3gradients_reshape_10_grad_mlcreshapegrad_reshape_107
3gradients_reshape_11_grad_mlcreshapegrad_reshape_117
3gradients_reshape_12_grad_mlcreshapegrad_reshape_12A
=gradients_transpose_1_grad_invertpermutation_transpose_1_perm;
7gradients_transpose_1_grad_mlctransposegrad_transpose_1A
=gradients_transpose_2_grad_invertpermutation_transpose_2_perm;
7gradients_transpose_2_grad_mlctransposegrad_transpose_2A
=gradients_transpose_3_grad_invertpermutation_transpose_3_perm;
7gradients_transpose_3_grad_mlctransposegrad_transpose_3A
=gradients_transpose_4_grad_invertpermutation_transpose_4_perm;
7gradients_transpose_4_grad_mlctransposegrad_transpose_4A
=gradients_transpose_5_grad_invertpermutation_transpose_5_perm;
7gradients_transpose_5_grad_mlctransposegrad_transpose_5A
=gradients_transpose_6_grad_invertpermutation_transpose_6_perm;
7gradients_transpose_6_grad_mlctransposegrad_transpose_63
/gradients_split_2_grad_concat_split_2_split_dim5
1gradients_split_grad_mlcsplitgrad_split_split_dim+
'gradients_split_grad_mlcsplitgrad_split9
5gradients_split_1_grad_mlcsplitgrad_split_1_split_dim/
+gradients_split_1_grad_mlcsplitgrad_split_11
-gradients_reshape_grad_mlcreshapegrad_reshape
identity

identity_1

identity_2

identity_3

identity_4?u
gradients/grad_ys_0Identityplaceholder*
T0*'
_output_shapes
:?????????2
gradients/grad_ys_0|
gradients/grad_ys_1Identityplaceholder_1*
T0*,
_output_shapes
:??????????2
gradients/grad_ys_1w
gradients/grad_ys_2Identityplaceholder_2*
T0*'
_output_shapes
:?????????2
gradients/grad_ys_2f
gradients/grad_ys_3Identityplaceholder_3*
T0*
_output_shapes
: 2
gradients/grad_ys_3?
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:2$
"gradients/strided_slice_grad/Shape?
6gradients/strided_slice_grad/MLCStridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*
valueB:
?????????28
6gradients/strided_slice_grad/MLCStridedSliceGrad/begin?
4gradients/strided_slice_grad/MLCStridedSliceGrad/endConst*
_output_shapes
:*
dtype0*
valueB: 26
4gradients/strided_slice_grad/MLCStridedSliceGrad/end?
8gradients/strided_slice_grad/MLCStridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*
valueB:2:
8gradients/strided_slice_grad/MLCStridedSliceGrad/strides?
0gradients/strided_slice_grad/MLCStridedSliceGradMLCStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0?gradients/strided_slice_grad/MLCStridedSliceGrad/begin:output:0=gradients/strided_slice_grad/MLCStridedSliceGrad/end:output:0Agradients/strided_slice_grad/MLCStridedSliceGrad/strides:output:0gradients/grad_ys_0:output:0>gradients_strided_slice_grad_mlcstridedslicegrad_strided_slice*
Index0*
T0*,
_output_shapes
:??????????*
shrink_axis_mask22
0gradients/strided_slice_grad/MLCStridedSliceGrad?
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:2.
,gradients/transpose_7_grad/InvertPermutation?
+gradients/transpose_7_grad/MLCTransposeGradMLCTransposeGradgradients/grad_ys_1:output:00gradients/transpose_7_grad/InvertPermutation:y:07gradients_transpose_7_grad_mlctransposegrad_transpose_7*
T0*,
_output_shapes
:??????????2-
+gradients/transpose_7_grad/MLCTransposeGrad?
gradients/Squeeze_grad/ShapeShape%gradients_squeeze_grad_shape_cudnnrnn*
T0*
_output_shapes
:2
gradients/Squeeze_grad/Shape?
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*+
_output_shapes
:?????????2 
gradients/Squeeze_grad/Reshape?
gradients/AddNAddN9gradients/strided_slice_grad/MLCStridedSliceGrad:output:0/gradients/transpose_7_grad/MLCTransposeGrad:y:0*
N*
T0*C
_class9
75loc:@gradients/strided_slice_grad/MLCStridedSliceGrad*,
_output_shapes
:??????????2
gradients/AddNy
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
: 2
gradients/zeros_like?
gradients/zeros_like_1	ZerosLikegradients_zeros_like_1_cudnnrnn*
T0*
_output_shapes
:2
gradients/zeros_like_1?
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat+gradients_strided_slice_grad_shape_cudnnrnn%gradients_squeeze_grad_shape_cudnnrnngradients_zeros_like_cudnnrnngradients/AddN:sum:0'gradients/Squeeze_grad/Reshape:output:0gradients/zeros_like:y:0gradients_zeros_like_1_cudnnrnn*
T0*L
_output_shapes:
8:??????????:?????????: :?*
rnn_modegru2*
(gradients/CudnnRNN_grad/CudnnRNNBackprop?
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:2,
*gradients/transpose_grad/InvertPermutation?
)gradients/transpose_grad/MLCTransposeGradMLCTransposeGrad9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:02gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose*
T0*,
_output_shapes
:??????????2+
)gradients/transpose_grad/MLCTransposeGrad?
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
:2!
gradients/ExpandDims_grad/Shape?
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*'
_output_shapes
:?????????2#
!gradients/ExpandDims_grad/Reshape?
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:?2
gradients/concat_grad/Shape|
gradients/concat_grad/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_grad/add/y?
gradients/concat_grad/addAddV2%gradients_concat_grad_add_concat_axis$gradients/concat_grad/add/y:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/add?
)gradients/concat_grad/strided_slice/stackPack%gradients_concat_grad_add_concat_axis*
N*
T0*
_output_shapes
:2+
)gradients/concat_grad/strided_slice/stack?
+gradients/concat_grad/strided_slice/stack_1Packgradients/concat_grad/add:z:0*
N*
T0*
_output_shapes
:2-
+gradients/concat_grad/strided_slice/stack_1?
+gradients/concat_grad/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+gradients/concat_grad/strided_slice/stack_2?
#gradients/concat_grad/strided_sliceStridedSlice$gradients/concat_grad/Shape:output:02gradients/concat_grad/strided_slice/stack:output:04gradients/concat_grad/strided_slice/stack_1:output:04gradients/concat_grad/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#gradients/concat_grad/strided_slice?
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:?2
gradients/concat_grad/Shape_1?
gradients/concat_grad/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_grad/add_1/y?
gradients/concat_grad/add_1AddV2%gradients_concat_grad_add_concat_axis&gradients/concat_grad/add_1/y:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/add_1?
+gradients/concat_grad/strided_slice_1/stackPack%gradients_concat_grad_add_concat_axis*
N*
T0*
_output_shapes
:2-
+gradients/concat_grad/strided_slice_1/stack?
-gradients/concat_grad/strided_slice_1/stack_1Packgradients/concat_grad/add_1:z:0*
N*
T0*
_output_shapes
:2/
-gradients/concat_grad/strided_slice_1/stack_1?
-gradients/concat_grad/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-gradients/concat_grad/strided_slice_1/stack_2?
%gradients/concat_grad/strided_slice_1StridedSlice&gradients/concat_grad/Shape_1:output:04gradients/concat_grad/strided_slice_1/stack:output:06gradients/concat_grad/strided_slice_1/stack_1:output:06gradients/concat_grad/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%gradients/concat_grad/strided_slice_1?
gradients/concat_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:?2
gradients/concat_grad/Shape_2?
gradients/concat_grad/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_grad/add_2/y?
gradients/concat_grad/add_2AddV2%gradients_concat_grad_add_concat_axis&gradients/concat_grad/add_2/y:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/add_2?
+gradients/concat_grad/strided_slice_2/stackPack%gradients_concat_grad_add_concat_axis*
N*
T0*
_output_shapes
:2-
+gradients/concat_grad/strided_slice_2/stack?
-gradients/concat_grad/strided_slice_2/stack_1Packgradients/concat_grad/add_2:z:0*
N*
T0*
_output_shapes
:2/
-gradients/concat_grad/strided_slice_2/stack_1?
-gradients/concat_grad/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-gradients/concat_grad/strided_slice_2/stack_2?
%gradients/concat_grad/strided_slice_2StridedSlice&gradients/concat_grad/Shape_2:output:04gradients/concat_grad/strided_slice_2/stack:output:06gradients/concat_grad/strided_slice_2/stack_1:output:06gradients/concat_grad/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%gradients/concat_grad/strided_slice_2?
gradients/concat_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:?2
gradients/concat_grad/Shape_3?
gradients/concat_grad/add_3/yConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_grad/add_3/y?
gradients/concat_grad/add_3AddV2%gradients_concat_grad_add_concat_axis&gradients/concat_grad/add_3/y:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/add_3?
+gradients/concat_grad/strided_slice_3/stackPack%gradients_concat_grad_add_concat_axis*
N*
T0*
_output_shapes
:2-
+gradients/concat_grad/strided_slice_3/stack?
-gradients/concat_grad/strided_slice_3/stack_1Packgradients/concat_grad/add_3:z:0*
N*
T0*
_output_shapes
:2/
-gradients/concat_grad/strided_slice_3/stack_1?
-gradients/concat_grad/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-gradients/concat_grad/strided_slice_3/stack_2?
%gradients/concat_grad/strided_slice_3StridedSlice&gradients/concat_grad/Shape_3:output:04gradients/concat_grad/strided_slice_3/stack:output:06gradients/concat_grad/strided_slice_3/stack_1:output:06gradients/concat_grad/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%gradients/concat_grad/strided_slice_3?
gradients/concat_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:?2
gradients/concat_grad/Shape_4?
gradients/concat_grad/add_4/yConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_grad/add_4/y?
gradients/concat_grad/add_4AddV2%gradients_concat_grad_add_concat_axis&gradients/concat_grad/add_4/y:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/add_4?
+gradients/concat_grad/strided_slice_4/stackPack%gradients_concat_grad_add_concat_axis*
N*
T0*
_output_shapes
:2-
+gradients/concat_grad/strided_slice_4/stack?
-gradients/concat_grad/strided_slice_4/stack_1Packgradients/concat_grad/add_4:z:0*
N*
T0*
_output_shapes
:2/
-gradients/concat_grad/strided_slice_4/stack_1?
-gradients/concat_grad/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-gradients/concat_grad/strided_slice_4/stack_2?
%gradients/concat_grad/strided_slice_4StridedSlice&gradients/concat_grad/Shape_4:output:04gradients/concat_grad/strided_slice_4/stack:output:06gradients/concat_grad/strided_slice_4/stack_1:output:06gradients/concat_grad/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%gradients/concat_grad/strided_slice_4?
gradients/concat_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:?2
gradients/concat_grad/Shape_5?
gradients/concat_grad/add_5/yConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_grad/add_5/y?
gradients/concat_grad/add_5AddV2%gradients_concat_grad_add_concat_axis&gradients/concat_grad/add_5/y:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/add_5?
+gradients/concat_grad/strided_slice_5/stackPack%gradients_concat_grad_add_concat_axis*
N*
T0*
_output_shapes
:2-
+gradients/concat_grad/strided_slice_5/stack?
-gradients/concat_grad/strided_slice_5/stack_1Packgradients/concat_grad/add_5:z:0*
N*
T0*
_output_shapes
:2/
-gradients/concat_grad/strided_slice_5/stack_1?
-gradients/concat_grad/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-gradients/concat_grad/strided_slice_5/stack_2?
%gradients/concat_grad/strided_slice_5StridedSlice&gradients/concat_grad/Shape_5:output:04gradients/concat_grad/strided_slice_5/stack:output:06gradients/concat_grad/strided_slice_5/stack_1:output:06gradients/concat_grad/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%gradients/concat_grad/strided_slice_5?
gradients/concat_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape_6?
gradients/concat_grad/add_6/yConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_grad/add_6/y?
gradients/concat_grad/add_6AddV2%gradients_concat_grad_add_concat_axis&gradients/concat_grad/add_6/y:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/add_6?
+gradients/concat_grad/strided_slice_6/stackPack%gradients_concat_grad_add_concat_axis*
N*
T0*
_output_shapes
:2-
+gradients/concat_grad/strided_slice_6/stack?
-gradients/concat_grad/strided_slice_6/stack_1Packgradients/concat_grad/add_6:z:0*
N*
T0*
_output_shapes
:2/
-gradients/concat_grad/strided_slice_6/stack_1?
-gradients/concat_grad/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-gradients/concat_grad/strided_slice_6/stack_2?
%gradients/concat_grad/strided_slice_6StridedSlice&gradients/concat_grad/Shape_6:output:04gradients/concat_grad/strided_slice_6/stack:output:06gradients/concat_grad/strided_slice_6/stack_1:output:06gradients/concat_grad/strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%gradients/concat_grad/strided_slice_6?
gradients/concat_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape_7?
gradients/concat_grad/add_7/yConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_grad/add_7/y?
gradients/concat_grad/add_7AddV2%gradients_concat_grad_add_concat_axis&gradients/concat_grad/add_7/y:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/add_7?
+gradients/concat_grad/strided_slice_7/stackPack%gradients_concat_grad_add_concat_axis*
N*
T0*
_output_shapes
:2-
+gradients/concat_grad/strided_slice_7/stack?
-gradients/concat_grad/strided_slice_7/stack_1Packgradients/concat_grad/add_7:z:0*
N*
T0*
_output_shapes
:2/
-gradients/concat_grad/strided_slice_7/stack_1?
-gradients/concat_grad/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-gradients/concat_grad/strided_slice_7/stack_2?
%gradients/concat_grad/strided_slice_7StridedSlice&gradients/concat_grad/Shape_7:output:04gradients/concat_grad/strided_slice_7/stack:output:06gradients/concat_grad/strided_slice_7/stack_1:output:06gradients/concat_grad/strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%gradients/concat_grad/strided_slice_7?
gradients/concat_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape_8?
gradients/concat_grad/add_8/yConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_grad/add_8/y?
gradients/concat_grad/add_8AddV2%gradients_concat_grad_add_concat_axis&gradients/concat_grad/add_8/y:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/add_8?
+gradients/concat_grad/strided_slice_8/stackPack%gradients_concat_grad_add_concat_axis*
N*
T0*
_output_shapes
:2-
+gradients/concat_grad/strided_slice_8/stack?
-gradients/concat_grad/strided_slice_8/stack_1Packgradients/concat_grad/add_8:z:0*
N*
T0*
_output_shapes
:2/
-gradients/concat_grad/strided_slice_8/stack_1?
-gradients/concat_grad/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-gradients/concat_grad/strided_slice_8/stack_2?
%gradients/concat_grad/strided_slice_8StridedSlice&gradients/concat_grad/Shape_8:output:04gradients/concat_grad/strided_slice_8/stack:output:06gradients/concat_grad/strided_slice_8/stack_1:output:06gradients/concat_grad/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%gradients/concat_grad/strided_slice_8?
gradients/concat_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape_9?
gradients/concat_grad/add_9/yConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_grad/add_9/y?
gradients/concat_grad/add_9AddV2%gradients_concat_grad_add_concat_axis&gradients/concat_grad/add_9/y:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/add_9?
+gradients/concat_grad/strided_slice_9/stackPack%gradients_concat_grad_add_concat_axis*
N*
T0*
_output_shapes
:2-
+gradients/concat_grad/strided_slice_9/stack?
-gradients/concat_grad/strided_slice_9/stack_1Packgradients/concat_grad/add_9:z:0*
N*
T0*
_output_shapes
:2/
-gradients/concat_grad/strided_slice_9/stack_1?
-gradients/concat_grad/strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-gradients/concat_grad/strided_slice_9/stack_2?
%gradients/concat_grad/strided_slice_9StridedSlice&gradients/concat_grad/Shape_9:output:04gradients/concat_grad/strided_slice_9/stack:output:06gradients/concat_grad/strided_slice_9/stack_1:output:06gradients/concat_grad/strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%gradients/concat_grad/strided_slice_9?
gradients/concat_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB:2 
gradients/concat_grad/Shape_10?
gradients/concat_grad/add_10/yConst*
_output_shapes
: *
dtype0*
value	B :2 
gradients/concat_grad/add_10/y?
gradients/concat_grad/add_10AddV2%gradients_concat_grad_add_concat_axis'gradients/concat_grad/add_10/y:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/add_10?
,gradients/concat_grad/strided_slice_10/stackPack%gradients_concat_grad_add_concat_axis*
N*
T0*
_output_shapes
:2.
,gradients/concat_grad/strided_slice_10/stack?
.gradients/concat_grad/strided_slice_10/stack_1Pack gradients/concat_grad/add_10:z:0*
N*
T0*
_output_shapes
:20
.gradients/concat_grad/strided_slice_10/stack_1?
.gradients/concat_grad/strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.gradients/concat_grad/strided_slice_10/stack_2?
&gradients/concat_grad/strided_slice_10StridedSlice'gradients/concat_grad/Shape_10:output:05gradients/concat_grad/strided_slice_10/stack:output:07gradients/concat_grad/strided_slice_10/stack_1:output:07gradients/concat_grad/strided_slice_10/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2(
&gradients/concat_grad/strided_slice_10?
gradients/concat_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB:2 
gradients/concat_grad/Shape_11?
gradients/concat_grad/add_11/yConst*
_output_shapes
: *
dtype0*
value	B :2 
gradients/concat_grad/add_11/y?
gradients/concat_grad/add_11AddV2%gradients_concat_grad_add_concat_axis'gradients/concat_grad/add_11/y:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/add_11?
,gradients/concat_grad/strided_slice_11/stackPack%gradients_concat_grad_add_concat_axis*
N*
T0*
_output_shapes
:2.
,gradients/concat_grad/strided_slice_11/stack?
.gradients/concat_grad/strided_slice_11/stack_1Pack gradients/concat_grad/add_11:z:0*
N*
T0*
_output_shapes
:20
.gradients/concat_grad/strided_slice_11/stack_1?
.gradients/concat_grad/strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.gradients/concat_grad/strided_slice_11/stack_2?
&gradients/concat_grad/strided_slice_11StridedSlice'gradients/concat_grad/Shape_11:output:05gradients/concat_grad/strided_slice_11/stack:output:07gradients/concat_grad/strided_slice_11/stack_1:output:07gradients/concat_grad/strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2(
&gradients/concat_grad/strided_slice_11?
gradients/concat_grad/packedPack,gradients/concat_grad/strided_slice:output:0.gradients/concat_grad/strided_slice_1:output:0.gradients/concat_grad/strided_slice_2:output:0.gradients/concat_grad/strided_slice_3:output:0.gradients/concat_grad/strided_slice_4:output:0.gradients/concat_grad/strided_slice_5:output:0.gradients/concat_grad/strided_slice_6:output:0.gradients/concat_grad/strided_slice_7:output:0.gradients/concat_grad/strided_slice_8:output:0.gradients/concat_grad/strided_slice_9:output:0/gradients/concat_grad/strided_slice_10:output:0/gradients/concat_grad/strided_slice_11:output:0*
N*
T0*
_output_shapes
:2
gradients/concat_grad/packed?
%gradients/concat_grad/MLCConcatV2GradMLCConcatV2Grad:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0%gradients/concat_grad/packed:output:0%gradients_concat_grad_add_concat_axis$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0&gradients/concat_grad/Shape_2:output:0&gradients/concat_grad/Shape_3:output:0&gradients/concat_grad/Shape_4:output:0&gradients/concat_grad/Shape_5:output:0&gradients/concat_grad/Shape_6:output:0&gradients/concat_grad/Shape_7:output:0&gradients/concat_grad/Shape_8:output:0&gradients/concat_grad/Shape_9:output:0'gradients/concat_grad/Shape_10:output:0'gradients/concat_grad/Shape_11:output:0/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat*
N*
T0*

Tlen0*b
_output_shapesP
N:?:?:?:?:?:?::::::2'
%gradients/concat_grad/MLCConcatV2Grad?
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
gradients/Reshape_1_grad/Shape?
'gradients/Reshape_1_grad/MLCReshapeGradMLCReshapeGrad.gradients/concat_grad/MLCConcatV2Grad:output:0'gradients/Reshape_1_grad/Shape:output:01gradients_reshape_1_grad_mlcreshapegrad_reshape_1*
T0*
_output_shapes

:2)
'gradients/Reshape_1_grad/MLCReshapeGrad?
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
gradients/Reshape_2_grad/Shape?
'gradients/Reshape_2_grad/MLCReshapeGradMLCReshapeGrad.gradients/concat_grad/MLCConcatV2Grad:output:1'gradients/Reshape_2_grad/Shape:output:01gradients_reshape_2_grad_mlcreshapegrad_reshape_2*
T0*
_output_shapes

:2)
'gradients/Reshape_2_grad/MLCReshapeGrad?
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
gradients/Reshape_3_grad/Shape?
'gradients/Reshape_3_grad/MLCReshapeGradMLCReshapeGrad.gradients/concat_grad/MLCConcatV2Grad:output:2'gradients/Reshape_3_grad/Shape:output:01gradients_reshape_3_grad_mlcreshapegrad_reshape_3*
T0*
_output_shapes

:2)
'gradients/Reshape_3_grad/MLCReshapeGrad?
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
gradients/Reshape_4_grad/Shape?
'gradients/Reshape_4_grad/MLCReshapeGradMLCReshapeGrad.gradients/concat_grad/MLCConcatV2Grad:output:3'gradients/Reshape_4_grad/Shape:output:01gradients_reshape_4_grad_mlcreshapegrad_reshape_4*
T0*
_output_shapes

:2)
'gradients/Reshape_4_grad/MLCReshapeGrad?
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
gradients/Reshape_5_grad/Shape?
'gradients/Reshape_5_grad/MLCReshapeGradMLCReshapeGrad.gradients/concat_grad/MLCConcatV2Grad:output:4'gradients/Reshape_5_grad/Shape:output:01gradients_reshape_5_grad_mlcreshapegrad_reshape_5*
T0*
_output_shapes

:2)
'gradients/Reshape_5_grad/MLCReshapeGrad?
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
gradients/Reshape_6_grad/Shape?
'gradients/Reshape_6_grad/MLCReshapeGradMLCReshapeGrad.gradients/concat_grad/MLCConcatV2Grad:output:5'gradients/Reshape_6_grad/Shape:output:01gradients_reshape_6_grad_mlcreshapegrad_reshape_6*
T0*
_output_shapes

:2)
'gradients/Reshape_6_grad/MLCReshapeGrad?
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2 
gradients/Reshape_7_grad/Shape?
'gradients/Reshape_7_grad/MLCReshapeGradMLCReshapeGrad.gradients/concat_grad/MLCConcatV2Grad:output:6'gradients/Reshape_7_grad/Shape:output:01gradients_reshape_7_grad_mlcreshapegrad_reshape_7*
T0*
_output_shapes
:2)
'gradients/Reshape_7_grad/MLCReshapeGrad?
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2 
gradients/Reshape_8_grad/Shape?
'gradients/Reshape_8_grad/MLCReshapeGradMLCReshapeGrad.gradients/concat_grad/MLCConcatV2Grad:output:7'gradients/Reshape_8_grad/Shape:output:01gradients_reshape_8_grad_mlcreshapegrad_reshape_8*
T0*
_output_shapes
:2)
'gradients/Reshape_8_grad/MLCReshapeGrad?
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2 
gradients/Reshape_9_grad/Shape?
'gradients/Reshape_9_grad/MLCReshapeGradMLCReshapeGrad.gradients/concat_grad/MLCConcatV2Grad:output:8'gradients/Reshape_9_grad/Shape:output:01gradients_reshape_9_grad_mlcreshapegrad_reshape_9*
T0*
_output_shapes
:2)
'gradients/Reshape_9_grad/MLCReshapeGrad?
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2!
gradients/Reshape_10_grad/Shape?
(gradients/Reshape_10_grad/MLCReshapeGradMLCReshapeGrad.gradients/concat_grad/MLCConcatV2Grad:output:9(gradients/Reshape_10_grad/Shape:output:03gradients_reshape_10_grad_mlcreshapegrad_reshape_10*
T0*
_output_shapes
:2*
(gradients/Reshape_10_grad/MLCReshapeGrad?
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2!
gradients/Reshape_11_grad/Shape?
(gradients/Reshape_11_grad/MLCReshapeGradMLCReshapeGrad/gradients/concat_grad/MLCConcatV2Grad:output:10(gradients/Reshape_11_grad/Shape:output:03gradients_reshape_11_grad_mlcreshapegrad_reshape_11*
T0*
_output_shapes
:2*
(gradients/Reshape_11_grad/MLCReshapeGrad?
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2!
gradients/Reshape_12_grad/Shape?
(gradients/Reshape_12_grad/MLCReshapeGradMLCReshapeGrad/gradients/concat_grad/MLCConcatV2Grad:output:11(gradients/Reshape_12_grad/Shape:output:03gradients_reshape_12_grad_mlcreshapegrad_reshape_12*
T0*
_output_shapes
:2*
(gradients/Reshape_12_grad/MLCReshapeGrad?
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:2.
,gradients/transpose_1_grad/InvertPermutation?
+gradients/transpose_1_grad/MLCTransposeGradMLCTransposeGrad0gradients/Reshape_1_grad/MLCReshapeGrad:output:00gradients/transpose_1_grad/InvertPermutation:y:07gradients_transpose_1_grad_mlctransposegrad_transpose_1*
T0*
_output_shapes

:2-
+gradients/transpose_1_grad/MLCTransposeGrad?
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:2.
,gradients/transpose_2_grad/InvertPermutation?
+gradients/transpose_2_grad/MLCTransposeGradMLCTransposeGrad0gradients/Reshape_2_grad/MLCReshapeGrad:output:00gradients/transpose_2_grad/InvertPermutation:y:07gradients_transpose_2_grad_mlctransposegrad_transpose_2*
T0*
_output_shapes

:2-
+gradients/transpose_2_grad/MLCTransposeGrad?
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:2.
,gradients/transpose_3_grad/InvertPermutation?
+gradients/transpose_3_grad/MLCTransposeGradMLCTransposeGrad0gradients/Reshape_3_grad/MLCReshapeGrad:output:00gradients/transpose_3_grad/InvertPermutation:y:07gradients_transpose_3_grad_mlctransposegrad_transpose_3*
T0*
_output_shapes

:2-
+gradients/transpose_3_grad/MLCTransposeGrad?
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:2.
,gradients/transpose_4_grad/InvertPermutation?
+gradients/transpose_4_grad/MLCTransposeGradMLCTransposeGrad0gradients/Reshape_4_grad/MLCReshapeGrad:output:00gradients/transpose_4_grad/InvertPermutation:y:07gradients_transpose_4_grad_mlctransposegrad_transpose_4*
T0*
_output_shapes

:2-
+gradients/transpose_4_grad/MLCTransposeGrad?
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:2.
,gradients/transpose_5_grad/InvertPermutation?
+gradients/transpose_5_grad/MLCTransposeGradMLCTransposeGrad0gradients/Reshape_5_grad/MLCReshapeGrad:output:00gradients/transpose_5_grad/InvertPermutation:y:07gradients_transpose_5_grad_mlctransposegrad_transpose_5*
T0*
_output_shapes

:2-
+gradients/transpose_5_grad/MLCTransposeGrad?
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:2.
,gradients/transpose_6_grad/InvertPermutation?
+gradients/transpose_6_grad/MLCTransposeGradMLCTransposeGrad0gradients/Reshape_6_grad/MLCReshapeGrad:output:00gradients/transpose_6_grad/InvertPermutation:y:07gradients_transpose_6_grad_mlctransposegrad_transpose_6*
T0*
_output_shapes

:2-
+gradients/transpose_6_grad/MLCTransposeGrad?
gradients/split_2_grad/concatConcatV20gradients/Reshape_7_grad/MLCReshapeGrad:output:00gradients/Reshape_8_grad/MLCReshapeGrad:output:00gradients/Reshape_9_grad/MLCReshapeGrad:output:01gradients/Reshape_10_grad/MLCReshapeGrad:output:01gradients/Reshape_11_grad/MLCReshapeGrad:output:01gradients/Reshape_12_grad/MLCReshapeGrad:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes	
:?2
gradients/split_2_grad/concat?
!gradients/split_grad/MLCSplitGradMLCSplitGrad/gradients/transpose_1_grad/MLCTransposeGrad:y:0/gradients/transpose_2_grad/MLCTransposeGrad:y:0/gradients/transpose_3_grad/MLCTransposeGrad:y:01gradients_split_grad_mlcsplitgrad_split_split_dim'gradients_split_grad_mlcsplitgrad_split*
N*
T0*
_output_shapes

:Z2#
!gradients/split_grad/MLCSplitGrad?
#gradients/split_1_grad/MLCSplitGradMLCSplitGrad/gradients/transpose_4_grad/MLCTransposeGrad:y:0/gradients/transpose_5_grad/MLCTransposeGrad:y:0/gradients/transpose_6_grad/MLCTransposeGrad:y:05gradients_split_1_grad_mlcsplitgrad_split_1_split_dim+gradients_split_1_grad_mlcsplitgrad_split_1*
N*
T0*
_output_shapes

:Z2%
#gradients/split_1_grad/MLCSplitGrad?
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"   Z   2
gradients/Reshape_grad/Shape?
%gradients/Reshape_grad/MLCReshapeGradMLCReshapeGrad&gradients/split_2_grad/concat:output:0%gradients/Reshape_grad/Shape:output:0-gradients_reshape_grad_mlcreshapegrad_reshape*
T0*
_output_shapes

:Z2'
%gradients/Reshape_grad/MLCReshapeGrad?
IdentityIdentity-gradients/transpose_grad/MLCTransposeGrad:y:0*
T0*,
_output_shapes
:??????????2

Identity?

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0*
T0*'
_output_shapes
:?????????2

Identity_1y

Identity_2Identity*gradients/split_grad/MLCSplitGrad:output:0*
T0*
_output_shapes

:Z2

Identity_2{

Identity_3Identity,gradients/split_1_grad/MLCSplitGrad:output:0*
T0*
_output_shapes

:Z2

Identity_3}

Identity_4Identity.gradients/Reshape_grad/MLCReshapeGrad:output:0*
T0*
_output_shapes

:Z2

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*?
_input_shapes?
?:?????????:??????????:?????????: :??????????:?????????::??????????:?????????: ::??????????:?????????: :?::?????????: :?:?:?:?:?:?::::::::::::::::::: : :: ::?*<
api_implements*(gru_b74c1e92-5c0b-403e-bc2f-dc8926149f3e*
api_preferred_deviceGPU*@
forward_function_name'%__forward_gpu_gru_with_fallback_14284*
go_backwards( *

time_major( :- )
'
_output_shapes
:?????????:2.
,
_output_shapes
:??????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :2.
,
_output_shapes
:??????????:-)
'
_output_shapes
:?????????: 

_output_shapes
::2.
,
_output_shapes
:??????????:1-
+
_output_shapes
:?????????:	

_output_shapes
: :


_output_shapes
::2.
,
_output_shapes
:??????????:1-
+
_output_shapes
:?????????:

_output_shapes
: :!

_output_shapes	
:?: 

_output_shapes
::-)
'
_output_shapes
:?????????:

_output_shapes
: :!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::$ 

_output_shapes

::  

_output_shapes
::$! 

_output_shapes

:: "

_output_shapes
::$# 

_output_shapes

:: $

_output_shapes
::$% 

_output_shapes

:: &

_output_shapes
::$' 

_output_shapes

:: (

_output_shapes
::$) 

_output_shapes

::*

_output_shapes
: :+

_output_shapes
: :$, 

_output_shapes

::-

_output_shapes
: :$. 

_output_shapes

::!/

_output_shapes	
:?
?E
?
__inference_standard_gru_15885

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3X
unstackUnpackbias*
T0* 
_output_shapes
:Z:Z*	
num2	
unstacku
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????2
	transposeK
ShapeShapetranspose:y:0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_1n
MatMulMatMulstrided_slice_1:output:0kernel*
T0*'
_output_shapes
:?????????Z2
MatMuls
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*'
_output_shapes
:?????????Z2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:?????????:?????????:?????????*
	num_split2
splitj
MatMul_1MatMulinit_hrecurrent_kernel*
T0*'
_output_shapes
:?????????Z2

MatMul_1y
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*'
_output_shapes
:?????????Z2
	BiasAdd_1T
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dim?
split_1Splitsplit_1/split_dim:output:0BiasAdd_1:output:0*
T0*M
_output_shapes;
9:?????????:?????????:?????????*
	num_split2	
split_1g
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:?????????2
addX
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:?????????2	
Sigmoidk
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:?????????2
add_1^
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:?????????2
	Sigmoid_1d
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:?????????2
mulb
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:?????????2
add_2Q
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:?????????2
Tanh\
mul_1MulSigmoid:y:0init_h*
T0*'
_output_shapes
:?????????2
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
sub/x`
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:?????????2
subZ
mul_2Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:?????????2
mul_2_
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:?????????2
add_3?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileStatelessWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelunstack:output:0recurrent_kernelunstack:output:1*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*S
_output_shapesA
?: : : : :?????????: : :Z:Z:Z:Z* 
_read_only_resource_inputs
 *
bodyR
while_body_15794*
condR
while_cond_15793*R
output_shapesA
?: : : : :?????????: : :Z:Z:Z:Z*
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_2y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  ??2	
runtimel
IdentityIdentitystrided_slice_2:output:0*
T0*'
_output_shapes
:?????????2

Identityt

Identity_1Identitytranspose_1:y:0*
T0*4
_output_shapes"
 :??????????????????2

Identity_1f

Identity_2Identitywhile:output:4*
T0*'
_output_shapes
:?????????2

Identity_2W

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: 2

Identity_3"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*d
_input_shapesS
Q:??????????????????:?????????:Z:Z:Z*<
api_implements*(gru_d320c0db-d27b-490e-8348-8d4a20a9c88e*
api_preferred_deviceCPU*
go_backwards( *

time_major( :\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinit_h:FB

_output_shapes

:Z
 
_user_specified_namekernel:PL

_output_shapes

:Z
*
_user_specified_namerecurrent_kernel:D@

_output_shapes

:Z

_user_specified_namebias
?<
?
'__inference_gpu_gru_with_fallback_17545

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3?u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm{
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:??????????2
	transposeb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims/dim}

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2

ExpandDimsP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
splitSplitsplit/split_dim:output:0kernel*
T0*2
_output_shapes 
:::*
	num_split2
splitT
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dim?
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*2
_output_shapes 
:::*
	num_split2	
split_1q
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
Reshape/shapea
ReshapeReshapebiasReshape/shape:output:0*
T0*
_output_shapes	
:?2	
ReshapeT
Const_2Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_2h
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_2/split_dim?
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*8
_output_shapes&
$::::::*
	num_split2	
split_2e
Const_3Const*
_output_shapes
:*
dtype0*
valueB:
?????????2	
Const_3u
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_1/perm{
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes

:2
transpose_1j
	Reshape_1Reshapetranspose_1:y:0Const_3:output:0*
T0*
_output_shapes	
:?2
	Reshape_1u
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_2/perm{
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes

:2
transpose_2j
	Reshape_2Reshapetranspose_2:y:0Const_3:output:0*
T0*
_output_shapes	
:?2
	Reshape_2u
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_3/perm{
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

:2
transpose_3j
	Reshape_3Reshapetranspose_3:y:0Const_3:output:0*
T0*
_output_shapes	
:?2
	Reshape_3u
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_4/perm}
transpose_4	Transposesplit_1:output:0transpose_4/perm:output:0*
T0*
_output_shapes

:2
transpose_4j
	Reshape_4Reshapetranspose_4:y:0Const_3:output:0*
T0*
_output_shapes	
:?2
	Reshape_4u
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_5/perm}
transpose_5	Transposesplit_1:output:1transpose_5/perm:output:0*
T0*
_output_shapes

:2
transpose_5j
	Reshape_5Reshapetranspose_5:y:0Const_3:output:0*
T0*
_output_shapes	
:?2
	Reshape_5u
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_6/perm}
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T0*
_output_shapes

:2
transpose_6j
	Reshape_6Reshapetranspose_6:y:0Const_3:output:0*
T0*
_output_shapes	
:?2
	Reshape_6j
	Reshape_7Reshapesplit_2:output:0Const_3:output:0*
T0*
_output_shapes
:2
	Reshape_7j
	Reshape_8Reshapesplit_2:output:1Const_3:output:0*
T0*
_output_shapes
:2
	Reshape_8j
	Reshape_9Reshapesplit_2:output:2Const_3:output:0*
T0*
_output_shapes
:2
	Reshape_9l

Reshape_10Reshapesplit_2:output:3Const_3:output:0*
T0*
_output_shapes
:2

Reshape_10l

Reshape_11Reshapesplit_2:output:4Const_3:output:0*
T0*
_output_shapes
:2

Reshape_11l

Reshape_12Reshapesplit_2:output:5Const_3:output:0*
T0*
_output_shapes
:2

Reshape_12\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat/axis?
concatConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
N*
T0*
_output_shapes	
:?2
concati
CudnnRNN/input_cConst*
_output_shapes
: *
dtype0*
valueB
 *    2
CudnnRNN/input_c?
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat:output:0*
T0*I
_output_shapes7
5:??????????:?????????: :*
rnn_modegru2

CudnnRNN}
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slicey
transpose_7/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_7/perm?
transpose_7	TransposeCudnnRNN:output:0transpose_7/perm:output:0*
T0*,
_output_shapes
:??????????2
transpose_7{
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:?????????*
squeeze_dims
 2	
Squeezef
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @2	
runtimej
IdentityIdentitystrided_slice:output:0*
T0*'
_output_shapes
:?????????2

Identityl

Identity_1Identitytranspose_7:y:0*
T0*,
_output_shapes
:??????????2

Identity_1h

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:?????????2

Identity_2W

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: 2

Identity_3"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*\
_input_shapesK
I:??????????:?????????:Z:Z:Z*<
api_implements*(gru_5f9dfa88-82d0-4c61-bd4c-33d58af38bad*
api_preferred_deviceGPU*
go_backwards( *

time_major( :T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinit_h:FB

_output_shapes

:Z
 
_user_specified_namekernel:PL

_output_shapes

:Z
*
_user_specified_namerecurrent_kernel:D@

_output_shapes

:Z

_user_specified_namebias
?

?
B__inference_dense_2_layer_call_and_return_conditional_losses_20210

inputs%
!mlcmatmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MLCMatMul/ReadVariableOp?
MLCMatMul/ReadVariableOpReadVariableOp!mlcmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MLCMatMul/ReadVariableOp?
	MLCMatMul	MLCMatMulinputs MLCMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
	MLCMatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMLCMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
TanhTanhBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Tanh?
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MLCMatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
MLCMatMul/ReadVariableOpMLCMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?=
?
'__inference_gpu_gru_with_fallback_15484

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3?u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????2
	transposeb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims/dim}

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2

ExpandDimsP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
splitSplitsplit/split_dim:output:0kernel*
T0*2
_output_shapes 
:::*
	num_split2
splitT
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dim?
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*2
_output_shapes 
:::*
	num_split2	
split_1q
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
Reshape/shapea
ReshapeReshapebiasReshape/shape:output:0*
T0*
_output_shapes	
:?2	
ReshapeT
Const_2Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_2h
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_2/split_dim?
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*8
_output_shapes&
$::::::*
	num_split2	
split_2e
Const_3Const*
_output_shapes
:*
dtype0*
valueB:
?????????2	
Const_3u
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_1/perm{
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes

:2
transpose_1j
	Reshape_1Reshapetranspose_1:y:0Const_3:output:0*
T0*
_output_shapes	
:?2
	Reshape_1u
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_2/perm{
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes

:2
transpose_2j
	Reshape_2Reshapetranspose_2:y:0Const_3:output:0*
T0*
_output_shapes	
:?2
	Reshape_2u
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_3/perm{
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

:2
transpose_3j
	Reshape_3Reshapetranspose_3:y:0Const_3:output:0*
T0*
_output_shapes	
:?2
	Reshape_3u
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_4/perm}
transpose_4	Transposesplit_1:output:0transpose_4/perm:output:0*
T0*
_output_shapes

:2
transpose_4j
	Reshape_4Reshapetranspose_4:y:0Const_3:output:0*
T0*
_output_shapes	
:?2
	Reshape_4u
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_5/perm}
transpose_5	Transposesplit_1:output:1transpose_5/perm:output:0*
T0*
_output_shapes

:2
transpose_5j
	Reshape_5Reshapetranspose_5:y:0Const_3:output:0*
T0*
_output_shapes	
:?2
	Reshape_5u
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_6/perm}
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T0*
_output_shapes

:2
transpose_6j
	Reshape_6Reshapetranspose_6:y:0Const_3:output:0*
T0*
_output_shapes	
:?2
	Reshape_6j
	Reshape_7Reshapesplit_2:output:0Const_3:output:0*
T0*
_output_shapes
:2
	Reshape_7j
	Reshape_8Reshapesplit_2:output:1Const_3:output:0*
T0*
_output_shapes
:2
	Reshape_8j
	Reshape_9Reshapesplit_2:output:2Const_3:output:0*
T0*
_output_shapes
:2
	Reshape_9l

Reshape_10Reshapesplit_2:output:3Const_3:output:0*
T0*
_output_shapes
:2

Reshape_10l

Reshape_11Reshapesplit_2:output:4Const_3:output:0*
T0*
_output_shapes
:2

Reshape_11l

Reshape_12Reshapesplit_2:output:5Const_3:output:0*
T0*
_output_shapes
:2

Reshape_12\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat/axis?
concatConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
N*
T0*
_output_shapes	
:?2
concati
CudnnRNN/input_cConst*
_output_shapes
: *
dtype0*
valueB
 *    2
CudnnRNN/input_c?
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat:output:0*
T0*Q
_output_shapes?
=:??????????????????:?????????: :*
rnn_modegru2

CudnnRNN}
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slicey
transpose_7/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_7/perm?
transpose_7	TransposeCudnnRNN:output:0transpose_7/perm:output:0*
T0*4
_output_shapes"
 :??????????????????2
transpose_7{
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:?????????*
squeeze_dims
 2	
Squeezef
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @2	
runtimej
IdentityIdentitystrided_slice:output:0*
T0*'
_output_shapes
:?????????2

Identityt

Identity_1Identitytranspose_7:y:0*
T0*4
_output_shapes"
 :??????????????????2

Identity_1h

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:?????????2

Identity_2W

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: 2

Identity_3"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*d
_input_shapesS
Q:??????????????????:?????????:Z:Z:Z*<
api_implements*(gru_85420d5a-5eb6-4eb6-bfb4-ef0c67d578e4*
api_preferred_deviceGPU*
go_backwards( *

time_major( :\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinit_h:FB

_output_shapes

:Z
 
_user_specified_namekernel:PL

_output_shapes

:Z
*
_user_specified_namerecurrent_kernel:D@

_output_shapes

:Z

_user_specified_namebias
?2
?
while_body_17851
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_matmul_kernel_0
while_biasadd_unstack_0%
!while_matmul_1_recurrent_kernel_0
while_biasadd_1_unstack_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_matmul_kernel
while_biasadd_unstack#
while_matmul_1_recurrent_kernel
while_biasadd_1_unstack?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
while/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while_matmul_kernel_0*
T0*'
_output_shapes
:?????????Z2
while/MatMul?
while/BiasAddBiasAddwhile/MatMul:product:0while_biasadd_unstack_0*
T0*'
_output_shapes
:?????????Z2
while/BiasAdd\
while/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/Constp
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
while/split/split_dim?
while/splitSplitwhile/split/split_dim:output:0while/BiasAdd:output:0*
T0*M
_output_shapes;
9:?????????:?????????:?????????*
	num_split2
while/split?
while/MatMul_1MatMulwhile_placeholder_2!while_matmul_1_recurrent_kernel_0*
T0*'
_output_shapes
:?????????Z2
while/MatMul_1?
while/BiasAdd_1BiasAddwhile/MatMul_1:product:0while_biasadd_1_unstack_0*
T0*'
_output_shapes
:?????????Z2
while/BiasAdd_1`
while/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
while/Const_1t
while/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
while/split_1/split_dim?
while/split_1Split while/split_1/split_dim:output:0while/BiasAdd_1:output:0*
T0*M
_output_shapes;
9:?????????:?????????:?????????*
	num_split2
while/split_1
	while/addAddV2while/split:output:0while/split_1:output:0*
T0*'
_output_shapes
:?????????2
	while/addj
while/SigmoidSigmoidwhile/add:z:0*
T0*'
_output_shapes
:?????????2
while/Sigmoid?
while/add_1AddV2while/split:output:1while/split_1:output:1*
T0*'
_output_shapes
:?????????2
while/add_1p
while/Sigmoid_1Sigmoidwhile/add_1:z:0*
T0*'
_output_shapes
:?????????2
while/Sigmoid_1|
	while/mulMulwhile/Sigmoid_1:y:0while/split_1:output:2*
T0*'
_output_shapes
:?????????2
	while/mulz
while/add_2AddV2while/split:output:2while/mul:z:0*
T0*'
_output_shapes
:?????????2
while/add_2c

while/TanhTanhwhile/add_2:z:0*
T0*'
_output_shapes
:?????????2

while/Tanh{
while/mul_1Mulwhile/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:?????????2
while/mul_1_
while/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
while/sub/xx
	while/subSubwhile/sub/x:output:0while/Sigmoid:y:0*
T0*'
_output_shapes
:?????????2
	while/subr
while/mul_2Mulwhile/sub:z:0while/Tanh:y:0*
T0*'
_output_shapes
:?????????2
while/mul_2w
while/add_3AddV2while/mul_1:z:0while/mul_2:z:0*
T0*'
_output_shapes
:?????????2
while/add_3?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/add_3:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem`
while/add_4/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_4/yo
while/add_4AddV2while_placeholderwhile/add_4/y:output:0*
T0*
_output_shapes
: 2
while/add_4`
while/add_5/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_5/yv
while/add_5AddV2while_while_loop_counterwhile/add_5/y:output:0*
T0*
_output_shapes
: 2
while/add_5^
while/IdentityIdentitywhile/add_5:z:0*
T0*
_output_shapes
: 2
while/Identityq
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: 2
while/Identity_1b
while/Identity_2Identitywhile/add_4:z:0*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3s
while/Identity_4Identitywhile/add_3:z:0*
T0*'
_output_shapes
:?????????2
while/Identity_4"4
while_biasadd_1_unstackwhile_biasadd_1_unstack_0"0
while_biasadd_unstackwhile_biasadd_unstack_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"D
while_matmul_1_recurrent_kernel!while_matmul_1_recurrent_kernel_0",
while_matmul_kernelwhile_matmul_kernel_0",
while_strided_slicewhile_strided_slice_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*R
_input_shapesA
?: : : : :?????????: : :Z:Z:Z:Z: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:Z: 

_output_shapes
:Z:$	 

_output_shapes

:Z: 


_output_shapes
:Z
?
?
G__inference_sequential_1_layer_call_and_return_conditional_losses_17228

inputs
	gru_17215
	gru_17217
	gru_17219
dense_2_17222
dense_2_17224
identity??dense_2/StatefulPartitionedCall?gru/StatefulPartitionedCall?
gru/StatefulPartitionedCallStatefulPartitionedCallinputs	gru_17215	gru_17217	gru_17219*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *G
fBR@
>__inference_gru_layer_call_and_return_conditional_losses_166662
gru/StatefulPartitionedCall?
dense_2/StatefulPartitionedCallStatefulPartitionedCall$gru/StatefulPartitionedCall:output:0dense_2_17222dense_2_17224*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_171762!
dense_2/StatefulPartitionedCall?
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0 ^dense_2/StatefulPartitionedCall^gru/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????:::::2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2:
gru/StatefulPartitionedCallgru/StatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
while_cond_19316
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice3
/while_while_cond_19316___redundant_placeholder03
/while_while_cond_19316___redundant_placeholder13
/while_while_cond_19316___redundant_placeholder23
/while_while_cond_19316___redundant_placeholder33
/while_while_cond_19316___redundant_placeholder4
while_identity
n

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*D
_input_shapes3
1: : : : :?????????: :::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
::

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
?E
?
__inference_standard_gru_16835

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3X
unstackUnpackbias*
T0* 
_output_shapes
:Z:Z*	
num2	
unstacku
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm{
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:??????????2
	transposeK
ShapeShapetranspose:y:0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_1n
MatMulMatMulstrided_slice_1:output:0kernel*
T0*'
_output_shapes
:?????????Z2
MatMuls
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*'
_output_shapes
:?????????Z2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:?????????:?????????:?????????*
	num_split2
splitj
MatMul_1MatMulinit_hrecurrent_kernel*
T0*'
_output_shapes
:?????????Z2

MatMul_1y
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*'
_output_shapes
:?????????Z2
	BiasAdd_1T
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dim?
split_1Splitsplit_1/split_dim:output:0BiasAdd_1:output:0*
T0*M
_output_shapes;
9:?????????:?????????:?????????*
	num_split2	
split_1g
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:?????????2
addX
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:?????????2	
Sigmoidk
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:?????????2
add_1^
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:?????????2
	Sigmoid_1d
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:?????????2
mulb
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:?????????2
add_2Q
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:?????????2
Tanh\
mul_1MulSigmoid:y:0init_h*
T0*'
_output_shapes
:?????????2
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
sub/x`
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:?????????2
subZ
mul_2Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:?????????2
mul_2_
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:?????????2
add_3?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileStatelessWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelunstack:output:0recurrent_kernelunstack:output:1*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*S
_output_shapesA
?: : : : :?????????: : :Z:Z:Z:Z* 
_read_only_resource_inputs
 *
bodyR
while_body_16744*
condR
while_cond_16743*R
output_shapesA
?: : : : :?????????: : :Z:Z:Z:Z*
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_2y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:??????????2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  ??2	
runtimel
IdentityIdentitystrided_slice_2:output:0*
T0*'
_output_shapes
:?????????2

Identityl

Identity_1Identitytranspose_1:y:0*
T0*,
_output_shapes
:??????????2

Identity_1f

Identity_2Identitywhile:output:4*
T0*'
_output_shapes
:?????????2

Identity_2W

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: 2

Identity_3"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*\
_input_shapesK
I:??????????:?????????:Z:Z:Z*<
api_implements*(gru_8d3b7182-ad7a-489b-86fa-c30b709435ce*
api_preferred_deviceCPU*
go_backwards( *

time_major( :T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinit_h:FB

_output_shapes

:Z
 
_user_specified_namekernel:PL

_output_shapes

:Z
*
_user_specified_namerecurrent_kernel:D@

_output_shapes

:Z

_user_specified_namebias
?
?
#__inference_gru_layer_call_fn_20188

inputs
unknown
	unknown_0
	unknown_1
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *G
fBR@
>__inference_gru_layer_call_and_return_conditional_losses_166662
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????:::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?M
?
%__forward_gpu_gru_with_fallback_14284

inputs
init_h_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3
cudnnrnn
strided_slice
transpose_7_perm
transpose_7

cudnnrnn_0

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
cudnnrnn_input_c

concat
transpose_perm

init_h
concat_axis
	reshape_1
	reshape_2
	reshape_3
	reshape_4
	reshape_5
	reshape_6
	reshape_7
	reshape_8
	reshape_9

reshape_10

reshape_11

reshape_12
transpose_1_perm
transpose_1
transpose_2_perm
transpose_2
transpose_3_perm
transpose_3
transpose_4_perm
transpose_4
transpose_5_perm
transpose_5
transpose_6_perm
transpose_6
split_2_split_dim
split_split_dim	
split
split_1_split_dim
split_1
reshape?u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permO
transpose_0	Transposeinputstranspose/perm:output:0*
T02
	transposeb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims/dim

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2

ExpandDimsP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dimU
split_0Splitsplit/split_dim:output:0kernel*
T0*
	num_split2
splitT
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dime
	split_1_0Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*
	num_split2	
split_1q
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
Reshape/shapea
ReshapeReshapebiasReshape/shape:output:0*
T0*
_output_shapes	
:?2	
ReshapeT
Const_2Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_2h
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_2/split_dim?
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*8
_output_shapes&
$::::::*
	num_split2	
split_2e
Const_3Const*
_output_shapes
:*
dtype0*
valueB:
?????????2	
Const_3u
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_1/perm_
transpose_1_0	Transposesplit_0:output:0transpose_1/perm:output:0*
T02
transpose_1l
	Reshape_1Reshapetranspose_1_0:y:0Const_3:output:0*
T0*
_output_shapes	
:?2
	Reshape_1u
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_2/perm_
transpose_2_0	Transposesplit_0:output:1transpose_2/perm:output:0*
T02
transpose_2l
	Reshape_2Reshapetranspose_2_0:y:0Const_3:output:0*
T0*
_output_shapes	
:?2
	Reshape_2u
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_3/perm_
transpose_3_0	Transposesplit_0:output:2transpose_3/perm:output:0*
T02
transpose_3l
	Reshape_3Reshapetranspose_3_0:y:0Const_3:output:0*
T0*
_output_shapes	
:?2
	Reshape_3u
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_4/perma
transpose_4_0	Transposesplit_1_0:output:0transpose_4/perm:output:0*
T02
transpose_4l
	Reshape_4Reshapetranspose_4_0:y:0Const_3:output:0*
T0*
_output_shapes	
:?2
	Reshape_4u
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_5/perma
transpose_5_0	Transposesplit_1_0:output:1transpose_5/perm:output:0*
T02
transpose_5l
	Reshape_5Reshapetranspose_5_0:y:0Const_3:output:0*
T0*
_output_shapes	
:?2
	Reshape_5u
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_6/perma
transpose_6_0	Transposesplit_1_0:output:2transpose_6/perm:output:0*
T02
transpose_6l
	Reshape_6Reshapetranspose_6_0:y:0Const_3:output:0*
T0*
_output_shapes	
:?2
	Reshape_6j
	Reshape_7Reshapesplit_2:output:0Const_3:output:0*
T0*
_output_shapes
:2
	Reshape_7j
	Reshape_8Reshapesplit_2:output:1Const_3:output:0*
T0*
_output_shapes
:2
	Reshape_8j
	Reshape_9Reshapesplit_2:output:2Const_3:output:0*
T0*
_output_shapes
:2
	Reshape_9l

Reshape_10Reshapesplit_2:output:3Const_3:output:0*
T0*
_output_shapes
:2

Reshape_10l

Reshape_11Reshapesplit_2:output:4Const_3:output:0*
T0*
_output_shapes
:2

Reshape_11l

Reshape_12Reshapesplit_2:output:5Const_3:output:0*
T0*
_output_shapes
:2

Reshape_12\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat/axis?
concat_0ConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
N*
T02
concati
CudnnRNN/input_cConst*
_output_shapes
: *
dtype0*
valueB
 *    2
CudnnRNN/input_c?
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat_0:output:0*
T0*I
_output_shapes7
5:??????????:?????????: :*
rnn_modegru2

CudnnRNN}
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_slice_0StridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
shrink_axis_mask2
strided_slicey
transpose_7/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_7/perm`
transpose_7_0	TransposeCudnnRNN:output:0transpose_7/perm:output:0*
T02
transpose_7{
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:?????????*
squeeze_dims
 2	
Squeezef
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @2	
runtimel
IdentityIdentitystrided_slice_0:output:0*
T0*'
_output_shapes
:?????????2

Identityn

Identity_1Identitytranspose_7_0:y:0*
T0*,
_output_shapes
:??????????2

Identity_1h

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:?????????2

Identity_2W

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: 2

Identity_3"
concatconcat_0:output:0"#
concat_axisconcat/axis:output:0"
cudnnrnnCudnnRNN:output:0"!

cudnnrnn_0CudnnRNN:output_h:0"!

cudnnrnn_1CudnnRNN:output_c:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"-
cudnnrnn_input_cCudnnRNN/input_c:output:0"!

expanddimsExpandDims:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"
init_hinit_h_0"
reshapeReshape:output:0"
	reshape_1Reshape_1:output:0"!

reshape_10Reshape_10:output:0"!

reshape_11Reshape_11:output:0"!

reshape_12Reshape_12:output:0"
	reshape_2Reshape_2:output:0"
	reshape_3Reshape_3:output:0"
	reshape_4Reshape_4:output:0"
	reshape_5Reshape_5:output:0"
	reshape_6Reshape_6:output:0"
	reshape_7Reshape_7:output:0"
	reshape_8Reshape_8:output:0"
	reshape_9Reshape_9:output:0"
splitsplit_0:output:0"
split_1split_1_0:output:0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0")
strided_slicestrided_slice_0:output:0"
	transposetranspose_0:y:0" 
transpose_1transpose_1_0:y:0"-
transpose_1_permtranspose_1/perm:output:0" 
transpose_2transpose_2_0:y:0"-
transpose_2_permtranspose_2/perm:output:0" 
transpose_3transpose_3_0:y:0"-
transpose_3_permtranspose_3/perm:output:0" 
transpose_4transpose_4_0:y:0"-
transpose_4_permtranspose_4/perm:output:0" 
transpose_5transpose_5_0:y:0"-
transpose_5_permtranspose_5/perm:output:0" 
transpose_6transpose_6_0:y:0"-
transpose_6_permtranspose_6/perm:output:0" 
transpose_7transpose_7_0:y:0"-
transpose_7_permtranspose_7/perm:output:0")
transpose_permtranspose/perm:output:0*\
_input_shapesK
I:??????????:?????????:Z:Z:Z*<
api_implements*(gru_b74c1e92-5c0b-403e-bc2f-dc8926149f3e*
api_preferred_deviceGPU*T
backward_function_name:8__inference___backward_gpu_gru_with_fallback_14067_14285*
go_backwards( *

time_major( :T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinit_h:FB

_output_shapes

:Z
 
_user_specified_namekernel:PL

_output_shapes

:Z
*
_user_specified_namerecurrent_kernel:D@

_output_shapes

:Z

_user_specified_namebias
?

?
B__inference_dense_2_layer_call_and_return_conditional_losses_17176

inputs%
!mlcmatmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MLCMatMul/ReadVariableOp?
MLCMatMul/ReadVariableOpReadVariableOp!mlcmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MLCMatMul/ReadVariableOp?
	MLCMatMul	MLCMatMulinputs MLCMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
	MLCMatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMLCMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
TanhTanhBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Tanh?
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MLCMatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
MLCMatMul/ReadVariableOpMLCMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
while_cond_16274
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice3
/while_while_cond_16274___redundant_placeholder03
/while_while_cond_16274___redundant_placeholder13
/while_while_cond_16274___redundant_placeholder23
/while_while_cond_16274___redundant_placeholder33
/while_while_cond_16274___redundant_placeholder4
while_identity
n

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*D
_input_shapes3
1: : : : :?????????: :::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
::

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
?E
?
__inference_standard_gru_17942

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3X
unstackUnpackbias*
T0* 
_output_shapes
:Z:Z*	
num2	
unstacku
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm{
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:??????????2
	transposeK
ShapeShapetranspose:y:0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_1n
MatMulMatMulstrided_slice_1:output:0kernel*
T0*'
_output_shapes
:?????????Z2
MatMuls
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*'
_output_shapes
:?????????Z2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:?????????:?????????:?????????*
	num_split2
splitj
MatMul_1MatMulinit_hrecurrent_kernel*
T0*'
_output_shapes
:?????????Z2

MatMul_1y
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*'
_output_shapes
:?????????Z2
	BiasAdd_1T
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dim?
split_1Splitsplit_1/split_dim:output:0BiasAdd_1:output:0*
T0*M
_output_shapes;
9:?????????:?????????:?????????*
	num_split2	
split_1g
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:?????????2
addX
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:?????????2	
Sigmoidk
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:?????????2
add_1^
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:?????????2
	Sigmoid_1d
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:?????????2
mulb
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:?????????2
add_2Q
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:?????????2
Tanh\
mul_1MulSigmoid:y:0init_h*
T0*'
_output_shapes
:?????????2
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
sub/x`
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:?????????2
subZ
mul_2Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:?????????2
mul_2_
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:?????????2
add_3?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileStatelessWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelunstack:output:0recurrent_kernelunstack:output:1*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*S
_output_shapesA
?: : : : :?????????: : :Z:Z:Z:Z* 
_read_only_resource_inputs
 *
bodyR
while_body_17851*
condR
while_cond_17850*R
output_shapesA
?: : : : :?????????: : :Z:Z:Z:Z*
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_2y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:??????????2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  ??2	
runtimel
IdentityIdentitystrided_slice_2:output:0*
T0*'
_output_shapes
:?????????2

Identityl

Identity_1Identitytranspose_1:y:0*
T0*,
_output_shapes
:??????????2

Identity_1f

Identity_2Identitywhile:output:4*
T0*'
_output_shapes
:?????????2

Identity_2W

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: 2

Identity_3"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*\
_input_shapesK
I:??????????:?????????:Z:Z:Z*<
api_implements*(gru_2db1bf7b-7db5-4f71-98ab-d0201f7417d2*
api_preferred_deviceCPU*
go_backwards( *

time_major( :T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinit_h:FB

_output_shapes

:Z
 
_user_specified_namekernel:PL

_output_shapes

:Z
*
_user_specified_namerecurrent_kernel:D@

_output_shapes

:Z

_user_specified_namebias
?2
?
while_body_18357
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_matmul_kernel_0
while_biasadd_unstack_0%
!while_matmul_1_recurrent_kernel_0
while_biasadd_1_unstack_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_matmul_kernel
while_biasadd_unstack#
while_matmul_1_recurrent_kernel
while_biasadd_1_unstack?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
while/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while_matmul_kernel_0*
T0*'
_output_shapes
:?????????Z2
while/MatMul?
while/BiasAddBiasAddwhile/MatMul:product:0while_biasadd_unstack_0*
T0*'
_output_shapes
:?????????Z2
while/BiasAdd\
while/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/Constp
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
while/split/split_dim?
while/splitSplitwhile/split/split_dim:output:0while/BiasAdd:output:0*
T0*M
_output_shapes;
9:?????????:?????????:?????????*
	num_split2
while/split?
while/MatMul_1MatMulwhile_placeholder_2!while_matmul_1_recurrent_kernel_0*
T0*'
_output_shapes
:?????????Z2
while/MatMul_1?
while/BiasAdd_1BiasAddwhile/MatMul_1:product:0while_biasadd_1_unstack_0*
T0*'
_output_shapes
:?????????Z2
while/BiasAdd_1`
while/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
while/Const_1t
while/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
while/split_1/split_dim?
while/split_1Split while/split_1/split_dim:output:0while/BiasAdd_1:output:0*
T0*M
_output_shapes;
9:?????????:?????????:?????????*
	num_split2
while/split_1
	while/addAddV2while/split:output:0while/split_1:output:0*
T0*'
_output_shapes
:?????????2
	while/addj
while/SigmoidSigmoidwhile/add:z:0*
T0*'
_output_shapes
:?????????2
while/Sigmoid?
while/add_1AddV2while/split:output:1while/split_1:output:1*
T0*'
_output_shapes
:?????????2
while/add_1p
while/Sigmoid_1Sigmoidwhile/add_1:z:0*
T0*'
_output_shapes
:?????????2
while/Sigmoid_1|
	while/mulMulwhile/Sigmoid_1:y:0while/split_1:output:2*
T0*'
_output_shapes
:?????????2
	while/mulz
while/add_2AddV2while/split:output:2while/mul:z:0*
T0*'
_output_shapes
:?????????2
while/add_2c

while/TanhTanhwhile/add_2:z:0*
T0*'
_output_shapes
:?????????2

while/Tanh{
while/mul_1Mulwhile/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:?????????2
while/mul_1_
while/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
while/sub/xx
	while/subSubwhile/sub/x:output:0while/Sigmoid:y:0*
T0*'
_output_shapes
:?????????2
	while/subr
while/mul_2Mulwhile/sub:z:0while/Tanh:y:0*
T0*'
_output_shapes
:?????????2
while/mul_2w
while/add_3AddV2while/mul_1:z:0while/mul_2:z:0*
T0*'
_output_shapes
:?????????2
while/add_3?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/add_3:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem`
while/add_4/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_4/yo
while/add_4AddV2while_placeholderwhile/add_4/y:output:0*
T0*
_output_shapes
: 2
while/add_4`
while/add_5/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_5/yv
while/add_5AddV2while_while_loop_counterwhile/add_5/y:output:0*
T0*
_output_shapes
: 2
while/add_5^
while/IdentityIdentitywhile/add_5:z:0*
T0*
_output_shapes
: 2
while/Identityq
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: 2
while/Identity_1b
while/Identity_2Identitywhile/add_4:z:0*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3s
while/Identity_4Identitywhile/add_3:z:0*
T0*'
_output_shapes
:?????????2
while/Identity_4"4
while_biasadd_1_unstackwhile_biasadd_1_unstack_0"0
while_biasadd_unstackwhile_biasadd_unstack_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"D
while_matmul_1_recurrent_kernel!while_matmul_1_recurrent_kernel_0",
while_matmul_kernelwhile_matmul_kernel_0",
while_strided_slicewhile_strided_slice_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*R
_input_shapesA
?: : : : :?????????: : :Z:Z:Z:Z: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:Z: 

_output_shapes
:Z:$	 

_output_shapes

:Z: 


_output_shapes
:Z
?<
?
'__inference_gpu_gru_with_fallback_14066

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3?u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm{
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:??????????2
	transposeb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims/dim}

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2

ExpandDimsP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
splitSplitsplit/split_dim:output:0kernel*
T0*2
_output_shapes 
:::*
	num_split2
splitT
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dim?
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*2
_output_shapes 
:::*
	num_split2	
split_1q
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
Reshape/shapea
ReshapeReshapebiasReshape/shape:output:0*
T0*
_output_shapes	
:?2	
ReshapeT
Const_2Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_2h
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_2/split_dim?
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*8
_output_shapes&
$::::::*
	num_split2	
split_2e
Const_3Const*
_output_shapes
:*
dtype0*
valueB:
?????????2	
Const_3u
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_1/perm{
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes

:2
transpose_1j
	Reshape_1Reshapetranspose_1:y:0Const_3:output:0*
T0*
_output_shapes	
:?2
	Reshape_1u
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_2/perm{
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes

:2
transpose_2j
	Reshape_2Reshapetranspose_2:y:0Const_3:output:0*
T0*
_output_shapes	
:?2
	Reshape_2u
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_3/perm{
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

:2
transpose_3j
	Reshape_3Reshapetranspose_3:y:0Const_3:output:0*
T0*
_output_shapes	
:?2
	Reshape_3u
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_4/perm}
transpose_4	Transposesplit_1:output:0transpose_4/perm:output:0*
T0*
_output_shapes

:2
transpose_4j
	Reshape_4Reshapetranspose_4:y:0Const_3:output:0*
T0*
_output_shapes	
:?2
	Reshape_4u
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_5/perm}
transpose_5	Transposesplit_1:output:1transpose_5/perm:output:0*
T0*
_output_shapes

:2
transpose_5j
	Reshape_5Reshapetranspose_5:y:0Const_3:output:0*
T0*
_output_shapes	
:?2
	Reshape_5u
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_6/perm}
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T0*
_output_shapes

:2
transpose_6j
	Reshape_6Reshapetranspose_6:y:0Const_3:output:0*
T0*
_output_shapes	
:?2
	Reshape_6j
	Reshape_7Reshapesplit_2:output:0Const_3:output:0*
T0*
_output_shapes
:2
	Reshape_7j
	Reshape_8Reshapesplit_2:output:1Const_3:output:0*
T0*
_output_shapes
:2
	Reshape_8j
	Reshape_9Reshapesplit_2:output:2Const_3:output:0*
T0*
_output_shapes
:2
	Reshape_9l

Reshape_10Reshapesplit_2:output:3Const_3:output:0*
T0*
_output_shapes
:2

Reshape_10l

Reshape_11Reshapesplit_2:output:4Const_3:output:0*
T0*
_output_shapes
:2

Reshape_11l

Reshape_12Reshapesplit_2:output:5Const_3:output:0*
T0*
_output_shapes
:2

Reshape_12\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat/axis?
concatConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
N*
T0*
_output_shapes	
:?2
concati
CudnnRNN/input_cConst*
_output_shapes
: *
dtype0*
valueB
 *    2
CudnnRNN/input_c?
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat:output:0*
T0*I
_output_shapes7
5:??????????:?????????: :*
rnn_modegru2

CudnnRNN}
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slicey
transpose_7/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_7/perm?
transpose_7	TransposeCudnnRNN:output:0transpose_7/perm:output:0*
T0*,
_output_shapes
:??????????2
transpose_7{
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:?????????*
squeeze_dims
 2	
Squeezef
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @2	
runtimej
IdentityIdentitystrided_slice:output:0*
T0*'
_output_shapes
:?????????2

Identityl

Identity_1Identitytranspose_7:y:0*
T0*,
_output_shapes
:??????????2

Identity_1h

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:?????????2

Identity_2W

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: 2

Identity_3"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*\
_input_shapesK
I:??????????:?????????:Z:Z:Z*<
api_implements*(gru_b74c1e92-5c0b-403e-bc2f-dc8926149f3e*
api_preferred_deviceGPU*
go_backwards( *

time_major( :T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinit_h:FB

_output_shapes

:Z
 
_user_specified_namekernel:PL

_output_shapes

:Z
*
_user_specified_namerecurrent_kernel:D@

_output_shapes

:Z

_user_specified_namebias
?M
?
%__forward_gpu_gru_with_fallback_16182

inputs
init_h_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3
cudnnrnn
strided_slice
transpose_7_perm
transpose_7

cudnnrnn_0

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
cudnnrnn_input_c

concat
transpose_perm

init_h
concat_axis
	reshape_1
	reshape_2
	reshape_3
	reshape_4
	reshape_5
	reshape_6
	reshape_7
	reshape_8
	reshape_9

reshape_10

reshape_11

reshape_12
transpose_1_perm
transpose_1
transpose_2_perm
transpose_2
transpose_3_perm
transpose_3
transpose_4_perm
transpose_4
transpose_5_perm
transpose_5
transpose_6_perm
transpose_6
split_2_split_dim
split_split_dim	
split
split_1_split_dim
split_1
reshape?u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permO
transpose_0	Transposeinputstranspose/perm:output:0*
T02
	transposeb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims/dim

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2

ExpandDimsP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dimU
split_0Splitsplit/split_dim:output:0kernel*
T0*
	num_split2
splitT
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dime
	split_1_0Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*
	num_split2	
split_1q
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
Reshape/shapea
ReshapeReshapebiasReshape/shape:output:0*
T0*
_output_shapes	
:?2	
ReshapeT
Const_2Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_2h
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_2/split_dim?
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*8
_output_shapes&
$::::::*
	num_split2	
split_2e
Const_3Const*
_output_shapes
:*
dtype0*
valueB:
?????????2	
Const_3u
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_1/perm_
transpose_1_0	Transposesplit_0:output:0transpose_1/perm:output:0*
T02
transpose_1l
	Reshape_1Reshapetranspose_1_0:y:0Const_3:output:0*
T0*
_output_shapes	
:?2
	Reshape_1u
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_2/perm_
transpose_2_0	Transposesplit_0:output:1transpose_2/perm:output:0*
T02
transpose_2l
	Reshape_2Reshapetranspose_2_0:y:0Const_3:output:0*
T0*
_output_shapes	
:?2
	Reshape_2u
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_3/perm_
transpose_3_0	Transposesplit_0:output:2transpose_3/perm:output:0*
T02
transpose_3l
	Reshape_3Reshapetranspose_3_0:y:0Const_3:output:0*
T0*
_output_shapes	
:?2
	Reshape_3u
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_4/perma
transpose_4_0	Transposesplit_1_0:output:0transpose_4/perm:output:0*
T02
transpose_4l
	Reshape_4Reshapetranspose_4_0:y:0Const_3:output:0*
T0*
_output_shapes	
:?2
	Reshape_4u
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_5/perma
transpose_5_0	Transposesplit_1_0:output:1transpose_5/perm:output:0*
T02
transpose_5l
	Reshape_5Reshapetranspose_5_0:y:0Const_3:output:0*
T0*
_output_shapes	
:?2
	Reshape_5u
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_6/perma
transpose_6_0	Transposesplit_1_0:output:2transpose_6/perm:output:0*
T02
transpose_6l
	Reshape_6Reshapetranspose_6_0:y:0Const_3:output:0*
T0*
_output_shapes	
:?2
	Reshape_6j
	Reshape_7Reshapesplit_2:output:0Const_3:output:0*
T0*
_output_shapes
:2
	Reshape_7j
	Reshape_8Reshapesplit_2:output:1Const_3:output:0*
T0*
_output_shapes
:2
	Reshape_8j
	Reshape_9Reshapesplit_2:output:2Const_3:output:0*
T0*
_output_shapes
:2
	Reshape_9l

Reshape_10Reshapesplit_2:output:3Const_3:output:0*
T0*
_output_shapes
:2

Reshape_10l

Reshape_11Reshapesplit_2:output:4Const_3:output:0*
T0*
_output_shapes
:2

Reshape_11l

Reshape_12Reshapesplit_2:output:5Const_3:output:0*
T0*
_output_shapes
:2

Reshape_12\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat/axis?
concat_0ConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
N*
T02
concati
CudnnRNN/input_cConst*
_output_shapes
: *
dtype0*
valueB
 *    2
CudnnRNN/input_c?
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat_0:output:0*
T0*Q
_output_shapes?
=:??????????????????:?????????: :*
rnn_modegru2

CudnnRNN}
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_slice_0StridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
shrink_axis_mask2
strided_slicey
transpose_7/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_7/perm`
transpose_7_0	TransposeCudnnRNN:output:0transpose_7/perm:output:0*
T02
transpose_7{
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:?????????*
squeeze_dims
 2	
Squeezef
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @2	
runtimel
IdentityIdentitystrided_slice_0:output:0*
T0*'
_output_shapes
:?????????2

Identityv

Identity_1Identitytranspose_7_0:y:0*
T0*4
_output_shapes"
 :??????????????????2

Identity_1h

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:?????????2

Identity_2W

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: 2

Identity_3"
concatconcat_0:output:0"#
concat_axisconcat/axis:output:0"
cudnnrnnCudnnRNN:output:0"!

cudnnrnn_0CudnnRNN:output_h:0"!

cudnnrnn_1CudnnRNN:output_c:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"-
cudnnrnn_input_cCudnnRNN/input_c:output:0"!

expanddimsExpandDims:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"
init_hinit_h_0"
reshapeReshape:output:0"
	reshape_1Reshape_1:output:0"!

reshape_10Reshape_10:output:0"!

reshape_11Reshape_11:output:0"!

reshape_12Reshape_12:output:0"
	reshape_2Reshape_2:output:0"
	reshape_3Reshape_3:output:0"
	reshape_4Reshape_4:output:0"
	reshape_5Reshape_5:output:0"
	reshape_6Reshape_6:output:0"
	reshape_7Reshape_7:output:0"
	reshape_8Reshape_8:output:0"
	reshape_9Reshape_9:output:0"
splitsplit_0:output:0"
split_1split_1_0:output:0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0")
strided_slicestrided_slice_0:output:0"
	transposetranspose_0:y:0" 
transpose_1transpose_1_0:y:0"-
transpose_1_permtranspose_1/perm:output:0" 
transpose_2transpose_2_0:y:0"-
transpose_2_permtranspose_2/perm:output:0" 
transpose_3transpose_3_0:y:0"-
transpose_3_permtranspose_3/perm:output:0" 
transpose_4transpose_4_0:y:0"-
transpose_4_permtranspose_4/perm:output:0" 
transpose_5transpose_5_0:y:0"-
transpose_5_permtranspose_5/perm:output:0" 
transpose_6transpose_6_0:y:0"-
transpose_6_permtranspose_6/perm:output:0" 
transpose_7transpose_7_0:y:0"-
transpose_7_permtranspose_7/perm:output:0")
transpose_permtranspose/perm:output:0*d
_input_shapesS
Q:??????????????????:?????????:Z:Z:Z*<
api_implements*(gru_d320c0db-d27b-490e-8348-8d4a20a9c88e*
api_preferred_deviceGPU*T
backward_function_name:8__inference___backward_gpu_gru_with_fallback_15965_16183*
go_backwards( *

time_major( :\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinit_h:FB

_output_shapes

:Z
 
_user_specified_namekernel:PL

_output_shapes

:Z
*
_user_specified_namerecurrent_kernel:D@

_output_shapes

:Z

_user_specified_namebias
?M
?
%__forward_gpu_gru_with_fallback_16663

inputs
init_h_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3
cudnnrnn
strided_slice
transpose_7_perm
transpose_7

cudnnrnn_0

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
cudnnrnn_input_c

concat
transpose_perm

init_h
concat_axis
	reshape_1
	reshape_2
	reshape_3
	reshape_4
	reshape_5
	reshape_6
	reshape_7
	reshape_8
	reshape_9

reshape_10

reshape_11

reshape_12
transpose_1_perm
transpose_1
transpose_2_perm
transpose_2
transpose_3_perm
transpose_3
transpose_4_perm
transpose_4
transpose_5_perm
transpose_5
transpose_6_perm
transpose_6
split_2_split_dim
split_split_dim	
split
split_1_split_dim
split_1
reshape?u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permO
transpose_0	Transposeinputstranspose/perm:output:0*
T02
	transposeb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims/dim

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2

ExpandDimsP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dimU
split_0Splitsplit/split_dim:output:0kernel*
T0*
	num_split2
splitT
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dime
	split_1_0Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*
	num_split2	
split_1q
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
Reshape/shapea
ReshapeReshapebiasReshape/shape:output:0*
T0*
_output_shapes	
:?2	
ReshapeT
Const_2Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_2h
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_2/split_dim?
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*8
_output_shapes&
$::::::*
	num_split2	
split_2e
Const_3Const*
_output_shapes
:*
dtype0*
valueB:
?????????2	
Const_3u
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_1/perm_
transpose_1_0	Transposesplit_0:output:0transpose_1/perm:output:0*
T02
transpose_1l
	Reshape_1Reshapetranspose_1_0:y:0Const_3:output:0*
T0*
_output_shapes	
:?2
	Reshape_1u
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_2/perm_
transpose_2_0	Transposesplit_0:output:1transpose_2/perm:output:0*
T02
transpose_2l
	Reshape_2Reshapetranspose_2_0:y:0Const_3:output:0*
T0*
_output_shapes	
:?2
	Reshape_2u
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_3/perm_
transpose_3_0	Transposesplit_0:output:2transpose_3/perm:output:0*
T02
transpose_3l
	Reshape_3Reshapetranspose_3_0:y:0Const_3:output:0*
T0*
_output_shapes	
:?2
	Reshape_3u
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_4/perma
transpose_4_0	Transposesplit_1_0:output:0transpose_4/perm:output:0*
T02
transpose_4l
	Reshape_4Reshapetranspose_4_0:y:0Const_3:output:0*
T0*
_output_shapes	
:?2
	Reshape_4u
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_5/perma
transpose_5_0	Transposesplit_1_0:output:1transpose_5/perm:output:0*
T02
transpose_5l
	Reshape_5Reshapetranspose_5_0:y:0Const_3:output:0*
T0*
_output_shapes	
:?2
	Reshape_5u
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_6/perma
transpose_6_0	Transposesplit_1_0:output:2transpose_6/perm:output:0*
T02
transpose_6l
	Reshape_6Reshapetranspose_6_0:y:0Const_3:output:0*
T0*
_output_shapes	
:?2
	Reshape_6j
	Reshape_7Reshapesplit_2:output:0Const_3:output:0*
T0*
_output_shapes
:2
	Reshape_7j
	Reshape_8Reshapesplit_2:output:1Const_3:output:0*
T0*
_output_shapes
:2
	Reshape_8j
	Reshape_9Reshapesplit_2:output:2Const_3:output:0*
T0*
_output_shapes
:2
	Reshape_9l

Reshape_10Reshapesplit_2:output:3Const_3:output:0*
T0*
_output_shapes
:2

Reshape_10l

Reshape_11Reshapesplit_2:output:4Const_3:output:0*
T0*
_output_shapes
:2

Reshape_11l

Reshape_12Reshapesplit_2:output:5Const_3:output:0*
T0*
_output_shapes
:2

Reshape_12\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat/axis?
concat_0ConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
N*
T02
concati
CudnnRNN/input_cConst*
_output_shapes
: *
dtype0*
valueB
 *    2
CudnnRNN/input_c?
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat_0:output:0*
T0*I
_output_shapes7
5:??????????:?????????: :*
rnn_modegru2

CudnnRNN}
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_slice_0StridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
shrink_axis_mask2
strided_slicey
transpose_7/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_7/perm`
transpose_7_0	TransposeCudnnRNN:output:0transpose_7/perm:output:0*
T02
transpose_7{
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:?????????*
squeeze_dims
 2	
Squeezef
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @2	
runtimel
IdentityIdentitystrided_slice_0:output:0*
T0*'
_output_shapes
:?????????2

Identityn

Identity_1Identitytranspose_7_0:y:0*
T0*,
_output_shapes
:??????????2

Identity_1h

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:?????????2

Identity_2W

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: 2

Identity_3"
concatconcat_0:output:0"#
concat_axisconcat/axis:output:0"
cudnnrnnCudnnRNN:output:0"!

cudnnrnn_0CudnnRNN:output_h:0"!

cudnnrnn_1CudnnRNN:output_c:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"-
cudnnrnn_input_cCudnnRNN/input_c:output:0"!

expanddimsExpandDims:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"
init_hinit_h_0"
reshapeReshape:output:0"
	reshape_1Reshape_1:output:0"!

reshape_10Reshape_10:output:0"!

reshape_11Reshape_11:output:0"!

reshape_12Reshape_12:output:0"
	reshape_2Reshape_2:output:0"
	reshape_3Reshape_3:output:0"
	reshape_4Reshape_4:output:0"
	reshape_5Reshape_5:output:0"
	reshape_6Reshape_6:output:0"
	reshape_7Reshape_7:output:0"
	reshape_8Reshape_8:output:0"
	reshape_9Reshape_9:output:0"
splitsplit_0:output:0"
split_1split_1_0:output:0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0")
strided_slicestrided_slice_0:output:0"
	transposetranspose_0:y:0" 
transpose_1transpose_1_0:y:0"-
transpose_1_permtranspose_1/perm:output:0" 
transpose_2transpose_2_0:y:0"-
transpose_2_permtranspose_2/perm:output:0" 
transpose_3transpose_3_0:y:0"-
transpose_3_permtranspose_3/perm:output:0" 
transpose_4transpose_4_0:y:0"-
transpose_4_permtranspose_4/perm:output:0" 
transpose_5transpose_5_0:y:0"-
transpose_5_permtranspose_5/perm:output:0" 
transpose_6transpose_6_0:y:0"-
transpose_6_permtranspose_6/perm:output:0" 
transpose_7transpose_7_0:y:0"-
transpose_7_permtranspose_7/perm:output:0")
transpose_permtranspose/perm:output:0*\
_input_shapesK
I:??????????:?????????:Z:Z:Z*<
api_implements*(gru_4820c462-f968-4106-b392-2a4aa0828265*
api_preferred_deviceGPU*T
backward_function_name:8__inference___backward_gpu_gru_with_fallback_16446_16664*
go_backwards( *

time_major( :T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinit_h:FB

_output_shapes

:Z
 
_user_specified_namekernel:PL

_output_shapes

:Z
*
_user_specified_namerecurrent_kernel:D@

_output_shapes

:Z

_user_specified_namebias
?	
?
while_cond_18356
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice3
/while_while_cond_18356___redundant_placeholder03
/while_while_cond_18356___redundant_placeholder13
/while_while_cond_18356___redundant_placeholder23
/while_while_cond_18356___redundant_placeholder33
/while_while_cond_18356___redundant_placeholder4
while_identity
n

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*D
_input_shapes3
1: : : : :?????????: :::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
::

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
ڮ
?
8__inference___backward_gpu_gru_with_fallback_18528_18746
placeholder
placeholder_1
placeholder_2
placeholder_3/
+gradients_strided_slice_grad_shape_cudnnrnnB
>gradients_strided_slice_grad_mlcstridedslicegrad_strided_sliceA
=gradients_transpose_7_grad_invertpermutation_transpose_7_perm;
7gradients_transpose_7_grad_mlctransposegrad_transpose_7)
%gradients_squeeze_grad_shape_cudnnrnn!
gradients_zeros_like_cudnnrnn#
gradients_zeros_like_1_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims=
9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c3
/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h)
%gradients_concat_grad_add_concat_axis5
1gradients_reshape_1_grad_mlcreshapegrad_reshape_15
1gradients_reshape_2_grad_mlcreshapegrad_reshape_25
1gradients_reshape_3_grad_mlcreshapegrad_reshape_35
1gradients_reshape_4_grad_mlcreshapegrad_reshape_45
1gradients_reshape_5_grad_mlcreshapegrad_reshape_55
1gradients_reshape_6_grad_mlcreshapegrad_reshape_65
1gradients_reshape_7_grad_mlcreshapegrad_reshape_75
1gradients_reshape_8_grad_mlcreshapegrad_reshape_85
1gradients_reshape_9_grad_mlcreshapegrad_reshape_97
3gradients_reshape_10_grad_mlcreshapegrad_reshape_107
3gradients_reshape_11_grad_mlcreshapegrad_reshape_117
3gradients_reshape_12_grad_mlcreshapegrad_reshape_12A
=gradients_transpose_1_grad_invertpermutation_transpose_1_perm;
7gradients_transpose_1_grad_mlctransposegrad_transpose_1A
=gradients_transpose_2_grad_invertpermutation_transpose_2_perm;
7gradients_transpose_2_grad_mlctransposegrad_transpose_2A
=gradients_transpose_3_grad_invertpermutation_transpose_3_perm;
7gradients_transpose_3_grad_mlctransposegrad_transpose_3A
=gradients_transpose_4_grad_invertpermutation_transpose_4_perm;
7gradients_transpose_4_grad_mlctransposegrad_transpose_4A
=gradients_transpose_5_grad_invertpermutation_transpose_5_perm;
7gradients_transpose_5_grad_mlctransposegrad_transpose_5A
=gradients_transpose_6_grad_invertpermutation_transpose_6_perm;
7gradients_transpose_6_grad_mlctransposegrad_transpose_63
/gradients_split_2_grad_concat_split_2_split_dim5
1gradients_split_grad_mlcsplitgrad_split_split_dim+
'gradients_split_grad_mlcsplitgrad_split9
5gradients_split_1_grad_mlcsplitgrad_split_1_split_dim/
+gradients_split_1_grad_mlcsplitgrad_split_11
-gradients_reshape_grad_mlcreshapegrad_reshape
identity

identity_1

identity_2

identity_3

identity_4?u
gradients/grad_ys_0Identityplaceholder*
T0*'
_output_shapes
:?????????2
gradients/grad_ys_0?
gradients/grad_ys_1Identityplaceholder_1*
T0*4
_output_shapes"
 :??????????????????2
gradients/grad_ys_1w
gradients/grad_ys_2Identityplaceholder_2*
T0*'
_output_shapes
:?????????2
gradients/grad_ys_2f
gradients/grad_ys_3Identityplaceholder_3*
T0*
_output_shapes
: 2
gradients/grad_ys_3?
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:2$
"gradients/strided_slice_grad/Shape?
6gradients/strided_slice_grad/MLCStridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*
valueB:
?????????28
6gradients/strided_slice_grad/MLCStridedSliceGrad/begin?
4gradients/strided_slice_grad/MLCStridedSliceGrad/endConst*
_output_shapes
:*
dtype0*
valueB: 26
4gradients/strided_slice_grad/MLCStridedSliceGrad/end?
8gradients/strided_slice_grad/MLCStridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*
valueB:2:
8gradients/strided_slice_grad/MLCStridedSliceGrad/strides?
0gradients/strided_slice_grad/MLCStridedSliceGradMLCStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0?gradients/strided_slice_grad/MLCStridedSliceGrad/begin:output:0=gradients/strided_slice_grad/MLCStridedSliceGrad/end:output:0Agradients/strided_slice_grad/MLCStridedSliceGrad/strides:output:0gradients/grad_ys_0:output:0>gradients_strided_slice_grad_mlcstridedslicegrad_strided_slice*
Index0*
T0*4
_output_shapes"
 :??????????????????*
shrink_axis_mask22
0gradients/strided_slice_grad/MLCStridedSliceGrad?
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:2.
,gradients/transpose_7_grad/InvertPermutation?
+gradients/transpose_7_grad/MLCTransposeGradMLCTransposeGradgradients/grad_ys_1:output:00gradients/transpose_7_grad/InvertPermutation:y:07gradients_transpose_7_grad_mlctransposegrad_transpose_7*
T0*4
_output_shapes"
 :??????????????????2-
+gradients/transpose_7_grad/MLCTransposeGrad?
gradients/Squeeze_grad/ShapeShape%gradients_squeeze_grad_shape_cudnnrnn*
T0*
_output_shapes
:2
gradients/Squeeze_grad/Shape?
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*+
_output_shapes
:?????????2 
gradients/Squeeze_grad/Reshape?
gradients/AddNAddN9gradients/strided_slice_grad/MLCStridedSliceGrad:output:0/gradients/transpose_7_grad/MLCTransposeGrad:y:0*
N*
T0*C
_class9
75loc:@gradients/strided_slice_grad/MLCStridedSliceGrad*4
_output_shapes"
 :??????????????????2
gradients/AddNy
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
: 2
gradients/zeros_like?
gradients/zeros_like_1	ZerosLikegradients_zeros_like_1_cudnnrnn*
T0*
_output_shapes
:2
gradients/zeros_like_1?
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat+gradients_strided_slice_grad_shape_cudnnrnn%gradients_squeeze_grad_shape_cudnnrnngradients_zeros_like_cudnnrnngradients/AddN:sum:0'gradients/Squeeze_grad/Reshape:output:0gradients/zeros_like:y:0gradients_zeros_like_1_cudnnrnn*
T0*T
_output_shapesB
@:??????????????????:?????????: :?*
rnn_modegru2*
(gradients/CudnnRNN_grad/CudnnRNNBackprop?
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:2,
*gradients/transpose_grad/InvertPermutation?
)gradients/transpose_grad/MLCTransposeGradMLCTransposeGrad9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:02gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose*
T0*4
_output_shapes"
 :??????????????????2+
)gradients/transpose_grad/MLCTransposeGrad?
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
:2!
gradients/ExpandDims_grad/Shape?
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*'
_output_shapes
:?????????2#
!gradients/ExpandDims_grad/Reshape?
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:?2
gradients/concat_grad/Shape|
gradients/concat_grad/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_grad/add/y?
gradients/concat_grad/addAddV2%gradients_concat_grad_add_concat_axis$gradients/concat_grad/add/y:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/add?
)gradients/concat_grad/strided_slice/stackPack%gradients_concat_grad_add_concat_axis*
N*
T0*
_output_shapes
:2+
)gradients/concat_grad/strided_slice/stack?
+gradients/concat_grad/strided_slice/stack_1Packgradients/concat_grad/add:z:0*
N*
T0*
_output_shapes
:2-
+gradients/concat_grad/strided_slice/stack_1?
+gradients/concat_grad/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+gradients/concat_grad/strided_slice/stack_2?
#gradients/concat_grad/strided_sliceStridedSlice$gradients/concat_grad/Shape:output:02gradients/concat_grad/strided_slice/stack:output:04gradients/concat_grad/strided_slice/stack_1:output:04gradients/concat_grad/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#gradients/concat_grad/strided_slice?
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:?2
gradients/concat_grad/Shape_1?
gradients/concat_grad/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_grad/add_1/y?
gradients/concat_grad/add_1AddV2%gradients_concat_grad_add_concat_axis&gradients/concat_grad/add_1/y:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/add_1?
+gradients/concat_grad/strided_slice_1/stackPack%gradients_concat_grad_add_concat_axis*
N*
T0*
_output_shapes
:2-
+gradients/concat_grad/strided_slice_1/stack?
-gradients/concat_grad/strided_slice_1/stack_1Packgradients/concat_grad/add_1:z:0*
N*
T0*
_output_shapes
:2/
-gradients/concat_grad/strided_slice_1/stack_1?
-gradients/concat_grad/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-gradients/concat_grad/strided_slice_1/stack_2?
%gradients/concat_grad/strided_slice_1StridedSlice&gradients/concat_grad/Shape_1:output:04gradients/concat_grad/strided_slice_1/stack:output:06gradients/concat_grad/strided_slice_1/stack_1:output:06gradients/concat_grad/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%gradients/concat_grad/strided_slice_1?
gradients/concat_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:?2
gradients/concat_grad/Shape_2?
gradients/concat_grad/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_grad/add_2/y?
gradients/concat_grad/add_2AddV2%gradients_concat_grad_add_concat_axis&gradients/concat_grad/add_2/y:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/add_2?
+gradients/concat_grad/strided_slice_2/stackPack%gradients_concat_grad_add_concat_axis*
N*
T0*
_output_shapes
:2-
+gradients/concat_grad/strided_slice_2/stack?
-gradients/concat_grad/strided_slice_2/stack_1Packgradients/concat_grad/add_2:z:0*
N*
T0*
_output_shapes
:2/
-gradients/concat_grad/strided_slice_2/stack_1?
-gradients/concat_grad/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-gradients/concat_grad/strided_slice_2/stack_2?
%gradients/concat_grad/strided_slice_2StridedSlice&gradients/concat_grad/Shape_2:output:04gradients/concat_grad/strided_slice_2/stack:output:06gradients/concat_grad/strided_slice_2/stack_1:output:06gradients/concat_grad/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%gradients/concat_grad/strided_slice_2?
gradients/concat_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:?2
gradients/concat_grad/Shape_3?
gradients/concat_grad/add_3/yConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_grad/add_3/y?
gradients/concat_grad/add_3AddV2%gradients_concat_grad_add_concat_axis&gradients/concat_grad/add_3/y:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/add_3?
+gradients/concat_grad/strided_slice_3/stackPack%gradients_concat_grad_add_concat_axis*
N*
T0*
_output_shapes
:2-
+gradients/concat_grad/strided_slice_3/stack?
-gradients/concat_grad/strided_slice_3/stack_1Packgradients/concat_grad/add_3:z:0*
N*
T0*
_output_shapes
:2/
-gradients/concat_grad/strided_slice_3/stack_1?
-gradients/concat_grad/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-gradients/concat_grad/strided_slice_3/stack_2?
%gradients/concat_grad/strided_slice_3StridedSlice&gradients/concat_grad/Shape_3:output:04gradients/concat_grad/strided_slice_3/stack:output:06gradients/concat_grad/strided_slice_3/stack_1:output:06gradients/concat_grad/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%gradients/concat_grad/strided_slice_3?
gradients/concat_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:?2
gradients/concat_grad/Shape_4?
gradients/concat_grad/add_4/yConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_grad/add_4/y?
gradients/concat_grad/add_4AddV2%gradients_concat_grad_add_concat_axis&gradients/concat_grad/add_4/y:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/add_4?
+gradients/concat_grad/strided_slice_4/stackPack%gradients_concat_grad_add_concat_axis*
N*
T0*
_output_shapes
:2-
+gradients/concat_grad/strided_slice_4/stack?
-gradients/concat_grad/strided_slice_4/stack_1Packgradients/concat_grad/add_4:z:0*
N*
T0*
_output_shapes
:2/
-gradients/concat_grad/strided_slice_4/stack_1?
-gradients/concat_grad/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-gradients/concat_grad/strided_slice_4/stack_2?
%gradients/concat_grad/strided_slice_4StridedSlice&gradients/concat_grad/Shape_4:output:04gradients/concat_grad/strided_slice_4/stack:output:06gradients/concat_grad/strided_slice_4/stack_1:output:06gradients/concat_grad/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%gradients/concat_grad/strided_slice_4?
gradients/concat_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:?2
gradients/concat_grad/Shape_5?
gradients/concat_grad/add_5/yConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_grad/add_5/y?
gradients/concat_grad/add_5AddV2%gradients_concat_grad_add_concat_axis&gradients/concat_grad/add_5/y:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/add_5?
+gradients/concat_grad/strided_slice_5/stackPack%gradients_concat_grad_add_concat_axis*
N*
T0*
_output_shapes
:2-
+gradients/concat_grad/strided_slice_5/stack?
-gradients/concat_grad/strided_slice_5/stack_1Packgradients/concat_grad/add_5:z:0*
N*
T0*
_output_shapes
:2/
-gradients/concat_grad/strided_slice_5/stack_1?
-gradients/concat_grad/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-gradients/concat_grad/strided_slice_5/stack_2?
%gradients/concat_grad/strided_slice_5StridedSlice&gradients/concat_grad/Shape_5:output:04gradients/concat_grad/strided_slice_5/stack:output:06gradients/concat_grad/strided_slice_5/stack_1:output:06gradients/concat_grad/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%gradients/concat_grad/strided_slice_5?
gradients/concat_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape_6?
gradients/concat_grad/add_6/yConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_grad/add_6/y?
gradients/concat_grad/add_6AddV2%gradients_concat_grad_add_concat_axis&gradients/concat_grad/add_6/y:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/add_6?
+gradients/concat_grad/strided_slice_6/stackPack%gradients_concat_grad_add_concat_axis*
N*
T0*
_output_shapes
:2-
+gradients/concat_grad/strided_slice_6/stack?
-gradients/concat_grad/strided_slice_6/stack_1Packgradients/concat_grad/add_6:z:0*
N*
T0*
_output_shapes
:2/
-gradients/concat_grad/strided_slice_6/stack_1?
-gradients/concat_grad/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-gradients/concat_grad/strided_slice_6/stack_2?
%gradients/concat_grad/strided_slice_6StridedSlice&gradients/concat_grad/Shape_6:output:04gradients/concat_grad/strided_slice_6/stack:output:06gradients/concat_grad/strided_slice_6/stack_1:output:06gradients/concat_grad/strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%gradients/concat_grad/strided_slice_6?
gradients/concat_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape_7?
gradients/concat_grad/add_7/yConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_grad/add_7/y?
gradients/concat_grad/add_7AddV2%gradients_concat_grad_add_concat_axis&gradients/concat_grad/add_7/y:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/add_7?
+gradients/concat_grad/strided_slice_7/stackPack%gradients_concat_grad_add_concat_axis*
N*
T0*
_output_shapes
:2-
+gradients/concat_grad/strided_slice_7/stack?
-gradients/concat_grad/strided_slice_7/stack_1Packgradients/concat_grad/add_7:z:0*
N*
T0*
_output_shapes
:2/
-gradients/concat_grad/strided_slice_7/stack_1?
-gradients/concat_grad/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-gradients/concat_grad/strided_slice_7/stack_2?
%gradients/concat_grad/strided_slice_7StridedSlice&gradients/concat_grad/Shape_7:output:04gradients/concat_grad/strided_slice_7/stack:output:06gradients/concat_grad/strided_slice_7/stack_1:output:06gradients/concat_grad/strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%gradients/concat_grad/strided_slice_7?
gradients/concat_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape_8?
gradients/concat_grad/add_8/yConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_grad/add_8/y?
gradients/concat_grad/add_8AddV2%gradients_concat_grad_add_concat_axis&gradients/concat_grad/add_8/y:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/add_8?
+gradients/concat_grad/strided_slice_8/stackPack%gradients_concat_grad_add_concat_axis*
N*
T0*
_output_shapes
:2-
+gradients/concat_grad/strided_slice_8/stack?
-gradients/concat_grad/strided_slice_8/stack_1Packgradients/concat_grad/add_8:z:0*
N*
T0*
_output_shapes
:2/
-gradients/concat_grad/strided_slice_8/stack_1?
-gradients/concat_grad/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-gradients/concat_grad/strided_slice_8/stack_2?
%gradients/concat_grad/strided_slice_8StridedSlice&gradients/concat_grad/Shape_8:output:04gradients/concat_grad/strided_slice_8/stack:output:06gradients/concat_grad/strided_slice_8/stack_1:output:06gradients/concat_grad/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%gradients/concat_grad/strided_slice_8?
gradients/concat_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape_9?
gradients/concat_grad/add_9/yConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_grad/add_9/y?
gradients/concat_grad/add_9AddV2%gradients_concat_grad_add_concat_axis&gradients/concat_grad/add_9/y:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/add_9?
+gradients/concat_grad/strided_slice_9/stackPack%gradients_concat_grad_add_concat_axis*
N*
T0*
_output_shapes
:2-
+gradients/concat_grad/strided_slice_9/stack?
-gradients/concat_grad/strided_slice_9/stack_1Packgradients/concat_grad/add_9:z:0*
N*
T0*
_output_shapes
:2/
-gradients/concat_grad/strided_slice_9/stack_1?
-gradients/concat_grad/strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-gradients/concat_grad/strided_slice_9/stack_2?
%gradients/concat_grad/strided_slice_9StridedSlice&gradients/concat_grad/Shape_9:output:04gradients/concat_grad/strided_slice_9/stack:output:06gradients/concat_grad/strided_slice_9/stack_1:output:06gradients/concat_grad/strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%gradients/concat_grad/strided_slice_9?
gradients/concat_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB:2 
gradients/concat_grad/Shape_10?
gradients/concat_grad/add_10/yConst*
_output_shapes
: *
dtype0*
value	B :2 
gradients/concat_grad/add_10/y?
gradients/concat_grad/add_10AddV2%gradients_concat_grad_add_concat_axis'gradients/concat_grad/add_10/y:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/add_10?
,gradients/concat_grad/strided_slice_10/stackPack%gradients_concat_grad_add_concat_axis*
N*
T0*
_output_shapes
:2.
,gradients/concat_grad/strided_slice_10/stack?
.gradients/concat_grad/strided_slice_10/stack_1Pack gradients/concat_grad/add_10:z:0*
N*
T0*
_output_shapes
:20
.gradients/concat_grad/strided_slice_10/stack_1?
.gradients/concat_grad/strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.gradients/concat_grad/strided_slice_10/stack_2?
&gradients/concat_grad/strided_slice_10StridedSlice'gradients/concat_grad/Shape_10:output:05gradients/concat_grad/strided_slice_10/stack:output:07gradients/concat_grad/strided_slice_10/stack_1:output:07gradients/concat_grad/strided_slice_10/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2(
&gradients/concat_grad/strided_slice_10?
gradients/concat_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB:2 
gradients/concat_grad/Shape_11?
gradients/concat_grad/add_11/yConst*
_output_shapes
: *
dtype0*
value	B :2 
gradients/concat_grad/add_11/y?
gradients/concat_grad/add_11AddV2%gradients_concat_grad_add_concat_axis'gradients/concat_grad/add_11/y:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/add_11?
,gradients/concat_grad/strided_slice_11/stackPack%gradients_concat_grad_add_concat_axis*
N*
T0*
_output_shapes
:2.
,gradients/concat_grad/strided_slice_11/stack?
.gradients/concat_grad/strided_slice_11/stack_1Pack gradients/concat_grad/add_11:z:0*
N*
T0*
_output_shapes
:20
.gradients/concat_grad/strided_slice_11/stack_1?
.gradients/concat_grad/strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.gradients/concat_grad/strided_slice_11/stack_2?
&gradients/concat_grad/strided_slice_11StridedSlice'gradients/concat_grad/Shape_11:output:05gradients/concat_grad/strided_slice_11/stack:output:07gradients/concat_grad/strided_slice_11/stack_1:output:07gradients/concat_grad/strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2(
&gradients/concat_grad/strided_slice_11?
gradients/concat_grad/packedPack,gradients/concat_grad/strided_slice:output:0.gradients/concat_grad/strided_slice_1:output:0.gradients/concat_grad/strided_slice_2:output:0.gradients/concat_grad/strided_slice_3:output:0.gradients/concat_grad/strided_slice_4:output:0.gradients/concat_grad/strided_slice_5:output:0.gradients/concat_grad/strided_slice_6:output:0.gradients/concat_grad/strided_slice_7:output:0.gradients/concat_grad/strided_slice_8:output:0.gradients/concat_grad/strided_slice_9:output:0/gradients/concat_grad/strided_slice_10:output:0/gradients/concat_grad/strided_slice_11:output:0*
N*
T0*
_output_shapes
:2
gradients/concat_grad/packed?
%gradients/concat_grad/MLCConcatV2GradMLCConcatV2Grad:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0%gradients/concat_grad/packed:output:0%gradients_concat_grad_add_concat_axis$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0&gradients/concat_grad/Shape_2:output:0&gradients/concat_grad/Shape_3:output:0&gradients/concat_grad/Shape_4:output:0&gradients/concat_grad/Shape_5:output:0&gradients/concat_grad/Shape_6:output:0&gradients/concat_grad/Shape_7:output:0&gradients/concat_grad/Shape_8:output:0&gradients/concat_grad/Shape_9:output:0'gradients/concat_grad/Shape_10:output:0'gradients/concat_grad/Shape_11:output:0/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat*
N*
T0*

Tlen0*b
_output_shapesP
N:?:?:?:?:?:?::::::2'
%gradients/concat_grad/MLCConcatV2Grad?
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
gradients/Reshape_1_grad/Shape?
'gradients/Reshape_1_grad/MLCReshapeGradMLCReshapeGrad.gradients/concat_grad/MLCConcatV2Grad:output:0'gradients/Reshape_1_grad/Shape:output:01gradients_reshape_1_grad_mlcreshapegrad_reshape_1*
T0*
_output_shapes

:2)
'gradients/Reshape_1_grad/MLCReshapeGrad?
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
gradients/Reshape_2_grad/Shape?
'gradients/Reshape_2_grad/MLCReshapeGradMLCReshapeGrad.gradients/concat_grad/MLCConcatV2Grad:output:1'gradients/Reshape_2_grad/Shape:output:01gradients_reshape_2_grad_mlcreshapegrad_reshape_2*
T0*
_output_shapes

:2)
'gradients/Reshape_2_grad/MLCReshapeGrad?
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
gradients/Reshape_3_grad/Shape?
'gradients/Reshape_3_grad/MLCReshapeGradMLCReshapeGrad.gradients/concat_grad/MLCConcatV2Grad:output:2'gradients/Reshape_3_grad/Shape:output:01gradients_reshape_3_grad_mlcreshapegrad_reshape_3*
T0*
_output_shapes

:2)
'gradients/Reshape_3_grad/MLCReshapeGrad?
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
gradients/Reshape_4_grad/Shape?
'gradients/Reshape_4_grad/MLCReshapeGradMLCReshapeGrad.gradients/concat_grad/MLCConcatV2Grad:output:3'gradients/Reshape_4_grad/Shape:output:01gradients_reshape_4_grad_mlcreshapegrad_reshape_4*
T0*
_output_shapes

:2)
'gradients/Reshape_4_grad/MLCReshapeGrad?
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
gradients/Reshape_5_grad/Shape?
'gradients/Reshape_5_grad/MLCReshapeGradMLCReshapeGrad.gradients/concat_grad/MLCConcatV2Grad:output:4'gradients/Reshape_5_grad/Shape:output:01gradients_reshape_5_grad_mlcreshapegrad_reshape_5*
T0*
_output_shapes

:2)
'gradients/Reshape_5_grad/MLCReshapeGrad?
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
gradients/Reshape_6_grad/Shape?
'gradients/Reshape_6_grad/MLCReshapeGradMLCReshapeGrad.gradients/concat_grad/MLCConcatV2Grad:output:5'gradients/Reshape_6_grad/Shape:output:01gradients_reshape_6_grad_mlcreshapegrad_reshape_6*
T0*
_output_shapes

:2)
'gradients/Reshape_6_grad/MLCReshapeGrad?
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2 
gradients/Reshape_7_grad/Shape?
'gradients/Reshape_7_grad/MLCReshapeGradMLCReshapeGrad.gradients/concat_grad/MLCConcatV2Grad:output:6'gradients/Reshape_7_grad/Shape:output:01gradients_reshape_7_grad_mlcreshapegrad_reshape_7*
T0*
_output_shapes
:2)
'gradients/Reshape_7_grad/MLCReshapeGrad?
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2 
gradients/Reshape_8_grad/Shape?
'gradients/Reshape_8_grad/MLCReshapeGradMLCReshapeGrad.gradients/concat_grad/MLCConcatV2Grad:output:7'gradients/Reshape_8_grad/Shape:output:01gradients_reshape_8_grad_mlcreshapegrad_reshape_8*
T0*
_output_shapes
:2)
'gradients/Reshape_8_grad/MLCReshapeGrad?
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2 
gradients/Reshape_9_grad/Shape?
'gradients/Reshape_9_grad/MLCReshapeGradMLCReshapeGrad.gradients/concat_grad/MLCConcatV2Grad:output:8'gradients/Reshape_9_grad/Shape:output:01gradients_reshape_9_grad_mlcreshapegrad_reshape_9*
T0*
_output_shapes
:2)
'gradients/Reshape_9_grad/MLCReshapeGrad?
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2!
gradients/Reshape_10_grad/Shape?
(gradients/Reshape_10_grad/MLCReshapeGradMLCReshapeGrad.gradients/concat_grad/MLCConcatV2Grad:output:9(gradients/Reshape_10_grad/Shape:output:03gradients_reshape_10_grad_mlcreshapegrad_reshape_10*
T0*
_output_shapes
:2*
(gradients/Reshape_10_grad/MLCReshapeGrad?
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2!
gradients/Reshape_11_grad/Shape?
(gradients/Reshape_11_grad/MLCReshapeGradMLCReshapeGrad/gradients/concat_grad/MLCConcatV2Grad:output:10(gradients/Reshape_11_grad/Shape:output:03gradients_reshape_11_grad_mlcreshapegrad_reshape_11*
T0*
_output_shapes
:2*
(gradients/Reshape_11_grad/MLCReshapeGrad?
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2!
gradients/Reshape_12_grad/Shape?
(gradients/Reshape_12_grad/MLCReshapeGradMLCReshapeGrad/gradients/concat_grad/MLCConcatV2Grad:output:11(gradients/Reshape_12_grad/Shape:output:03gradients_reshape_12_grad_mlcreshapegrad_reshape_12*
T0*
_output_shapes
:2*
(gradients/Reshape_12_grad/MLCReshapeGrad?
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:2.
,gradients/transpose_1_grad/InvertPermutation?
+gradients/transpose_1_grad/MLCTransposeGradMLCTransposeGrad0gradients/Reshape_1_grad/MLCReshapeGrad:output:00gradients/transpose_1_grad/InvertPermutation:y:07gradients_transpose_1_grad_mlctransposegrad_transpose_1*
T0*
_output_shapes

:2-
+gradients/transpose_1_grad/MLCTransposeGrad?
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:2.
,gradients/transpose_2_grad/InvertPermutation?
+gradients/transpose_2_grad/MLCTransposeGradMLCTransposeGrad0gradients/Reshape_2_grad/MLCReshapeGrad:output:00gradients/transpose_2_grad/InvertPermutation:y:07gradients_transpose_2_grad_mlctransposegrad_transpose_2*
T0*
_output_shapes

:2-
+gradients/transpose_2_grad/MLCTransposeGrad?
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:2.
,gradients/transpose_3_grad/InvertPermutation?
+gradients/transpose_3_grad/MLCTransposeGradMLCTransposeGrad0gradients/Reshape_3_grad/MLCReshapeGrad:output:00gradients/transpose_3_grad/InvertPermutation:y:07gradients_transpose_3_grad_mlctransposegrad_transpose_3*
T0*
_output_shapes

:2-
+gradients/transpose_3_grad/MLCTransposeGrad?
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:2.
,gradients/transpose_4_grad/InvertPermutation?
+gradients/transpose_4_grad/MLCTransposeGradMLCTransposeGrad0gradients/Reshape_4_grad/MLCReshapeGrad:output:00gradients/transpose_4_grad/InvertPermutation:y:07gradients_transpose_4_grad_mlctransposegrad_transpose_4*
T0*
_output_shapes

:2-
+gradients/transpose_4_grad/MLCTransposeGrad?
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:2.
,gradients/transpose_5_grad/InvertPermutation?
+gradients/transpose_5_grad/MLCTransposeGradMLCTransposeGrad0gradients/Reshape_5_grad/MLCReshapeGrad:output:00gradients/transpose_5_grad/InvertPermutation:y:07gradients_transpose_5_grad_mlctransposegrad_transpose_5*
T0*
_output_shapes

:2-
+gradients/transpose_5_grad/MLCTransposeGrad?
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:2.
,gradients/transpose_6_grad/InvertPermutation?
+gradients/transpose_6_grad/MLCTransposeGradMLCTransposeGrad0gradients/Reshape_6_grad/MLCReshapeGrad:output:00gradients/transpose_6_grad/InvertPermutation:y:07gradients_transpose_6_grad_mlctransposegrad_transpose_6*
T0*
_output_shapes

:2-
+gradients/transpose_6_grad/MLCTransposeGrad?
gradients/split_2_grad/concatConcatV20gradients/Reshape_7_grad/MLCReshapeGrad:output:00gradients/Reshape_8_grad/MLCReshapeGrad:output:00gradients/Reshape_9_grad/MLCReshapeGrad:output:01gradients/Reshape_10_grad/MLCReshapeGrad:output:01gradients/Reshape_11_grad/MLCReshapeGrad:output:01gradients/Reshape_12_grad/MLCReshapeGrad:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes	
:?2
gradients/split_2_grad/concat?
!gradients/split_grad/MLCSplitGradMLCSplitGrad/gradients/transpose_1_grad/MLCTransposeGrad:y:0/gradients/transpose_2_grad/MLCTransposeGrad:y:0/gradients/transpose_3_grad/MLCTransposeGrad:y:01gradients_split_grad_mlcsplitgrad_split_split_dim'gradients_split_grad_mlcsplitgrad_split*
N*
T0*
_output_shapes

:Z2#
!gradients/split_grad/MLCSplitGrad?
#gradients/split_1_grad/MLCSplitGradMLCSplitGrad/gradients/transpose_4_grad/MLCTransposeGrad:y:0/gradients/transpose_5_grad/MLCTransposeGrad:y:0/gradients/transpose_6_grad/MLCTransposeGrad:y:05gradients_split_1_grad_mlcsplitgrad_split_1_split_dim+gradients_split_1_grad_mlcsplitgrad_split_1*
N*
T0*
_output_shapes

:Z2%
#gradients/split_1_grad/MLCSplitGrad?
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"   Z   2
gradients/Reshape_grad/Shape?
%gradients/Reshape_grad/MLCReshapeGradMLCReshapeGrad&gradients/split_2_grad/concat:output:0%gradients/Reshape_grad/Shape:output:0-gradients_reshape_grad_mlcreshapegrad_reshape*
T0*
_output_shapes

:Z2'
%gradients/Reshape_grad/MLCReshapeGrad?
IdentityIdentity-gradients/transpose_grad/MLCTransposeGrad:y:0*
T0*4
_output_shapes"
 :??????????????????2

Identity?

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0*
T0*'
_output_shapes
:?????????2

Identity_1y

Identity_2Identity*gradients/split_grad/MLCSplitGrad:output:0*
T0*
_output_shapes

:Z2

Identity_2{

Identity_3Identity,gradients/split_1_grad/MLCSplitGrad:output:0*
T0*
_output_shapes

:Z2

Identity_3}

Identity_4Identity.gradients/Reshape_grad/MLCReshapeGrad:output:0*
T0*
_output_shapes

:Z2

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*?
_input_shapes?
?:?????????:??????????????????:?????????: :??????????????????:?????????::??????????????????:?????????: ::??????????????????:?????????: :?::?????????: :?:?:?:?:?:?::::::::::::::::::: : :: ::?*<
api_implements*(gru_7fd55799-5a12-43e2-bc93-7077640e6183*
api_preferred_deviceGPU*@
forward_function_name'%__forward_gpu_gru_with_fallback_18745*
go_backwards( *

time_major( :- )
'
_output_shapes
:?????????::6
4
_output_shapes"
 :??????????????????:-)
'
_output_shapes
:?????????:

_output_shapes
: ::6
4
_output_shapes"
 :??????????????????:-)
'
_output_shapes
:?????????: 

_output_shapes
:::6
4
_output_shapes"
 :??????????????????:1-
+
_output_shapes
:?????????:	

_output_shapes
: :


_output_shapes
:::6
4
_output_shapes"
 :??????????????????:1-
+
_output_shapes
:?????????:

_output_shapes
: :!

_output_shapes	
:?: 

_output_shapes
::-)
'
_output_shapes
:?????????:

_output_shapes
: :!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::$ 

_output_shapes

::  

_output_shapes
::$! 

_output_shapes

:: "

_output_shapes
::$# 

_output_shapes

:: $

_output_shapes
::$% 

_output_shapes

:: &

_output_shapes
::$' 

_output_shapes

:: (

_output_shapes
::$) 

_output_shapes

::*

_output_shapes
: :+

_output_shapes
: :$, 

_output_shapes

::-

_output_shapes
: :$. 

_output_shapes

::!/

_output_shapes	
:?
?	
?
while_cond_19785
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice3
/while_while_cond_19785___redundant_placeholder03
/while_while_cond_19785___redundant_placeholder13
/while_while_cond_19785___redundant_placeholder23
/while_while_cond_19785___redundant_placeholder33
/while_while_cond_19785___redundant_placeholder4
while_identity
n

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*D
_input_shapes3
1: : : : :?????????: :::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
::

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
?
|
'__inference_dense_2_layer_call_fn_20219

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_171762
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
#__inference_gru_layer_call_fn_19239
inputs_0
unknown
	unknown_0
	unknown_1
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *G
fBR@
>__inference_gru_layer_call_and_return_conditional_losses_161852
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????:::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :??????????????????
"
_user_specified_name
inputs/0
?E
?
__inference_standard_gru_16366

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3X
unstackUnpackbias*
T0* 
_output_shapes
:Z:Z*	
num2	
unstacku
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm{
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:??????????2
	transposeK
ShapeShapetranspose:y:0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_1n
MatMulMatMulstrided_slice_1:output:0kernel*
T0*'
_output_shapes
:?????????Z2
MatMuls
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*'
_output_shapes
:?????????Z2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:?????????:?????????:?????????*
	num_split2
splitj
MatMul_1MatMulinit_hrecurrent_kernel*
T0*'
_output_shapes
:?????????Z2

MatMul_1y
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*'
_output_shapes
:?????????Z2
	BiasAdd_1T
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dim?
split_1Splitsplit_1/split_dim:output:0BiasAdd_1:output:0*
T0*M
_output_shapes;
9:?????????:?????????:?????????*
	num_split2	
split_1g
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:?????????2
addX
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:?????????2	
Sigmoidk
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:?????????2
add_1^
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:?????????2
	Sigmoid_1d
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:?????????2
mulb
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:?????????2
add_2Q
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:?????????2
Tanh\
mul_1MulSigmoid:y:0init_h*
T0*'
_output_shapes
:?????????2
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
sub/x`
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:?????????2
subZ
mul_2Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:?????????2
mul_2_
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:?????????2
add_3?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileStatelessWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelunstack:output:0recurrent_kernelunstack:output:1*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*S
_output_shapesA
?: : : : :?????????: : :Z:Z:Z:Z* 
_read_only_resource_inputs
 *
bodyR
while_body_16275*
condR
while_cond_16274*R
output_shapesA
?: : : : :?????????: : :Z:Z:Z:Z*
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_2y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:??????????2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  ??2	
runtimel
IdentityIdentitystrided_slice_2:output:0*
T0*'
_output_shapes
:?????????2

Identityl

Identity_1Identitytranspose_1:y:0*
T0*,
_output_shapes
:??????????2

Identity_1f

Identity_2Identitywhile:output:4*
T0*'
_output_shapes
:?????????2

Identity_2W

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: 2

Identity_3"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*\
_input_shapesK
I:??????????:?????????:Z:Z:Z*<
api_implements*(gru_4820c462-f968-4106-b392-2a4aa0828265*
api_preferred_deviceCPU*
go_backwards( *

time_major( :T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinit_h:FB

_output_shapes

:Z
 
_user_specified_namekernel:PL

_output_shapes

:Z
*
_user_specified_namerecurrent_kernel:D@

_output_shapes

:Z

_user_specified_namebias
?=
?
'__inference_gpu_gru_with_fallback_15964

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3?u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????2
	transposeb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims/dim}

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2

ExpandDimsP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
splitSplitsplit/split_dim:output:0kernel*
T0*2
_output_shapes 
:::*
	num_split2
splitT
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dim?
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*2
_output_shapes 
:::*
	num_split2	
split_1q
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
Reshape/shapea
ReshapeReshapebiasReshape/shape:output:0*
T0*
_output_shapes	
:?2	
ReshapeT
Const_2Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_2h
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_2/split_dim?
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*8
_output_shapes&
$::::::*
	num_split2	
split_2e
Const_3Const*
_output_shapes
:*
dtype0*
valueB:
?????????2	
Const_3u
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_1/perm{
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes

:2
transpose_1j
	Reshape_1Reshapetranspose_1:y:0Const_3:output:0*
T0*
_output_shapes	
:?2
	Reshape_1u
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_2/perm{
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes

:2
transpose_2j
	Reshape_2Reshapetranspose_2:y:0Const_3:output:0*
T0*
_output_shapes	
:?2
	Reshape_2u
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_3/perm{
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

:2
transpose_3j
	Reshape_3Reshapetranspose_3:y:0Const_3:output:0*
T0*
_output_shapes	
:?2
	Reshape_3u
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_4/perm}
transpose_4	Transposesplit_1:output:0transpose_4/perm:output:0*
T0*
_output_shapes

:2
transpose_4j
	Reshape_4Reshapetranspose_4:y:0Const_3:output:0*
T0*
_output_shapes	
:?2
	Reshape_4u
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_5/perm}
transpose_5	Transposesplit_1:output:1transpose_5/perm:output:0*
T0*
_output_shapes

:2
transpose_5j
	Reshape_5Reshapetranspose_5:y:0Const_3:output:0*
T0*
_output_shapes	
:?2
	Reshape_5u
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_6/perm}
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T0*
_output_shapes

:2
transpose_6j
	Reshape_6Reshapetranspose_6:y:0Const_3:output:0*
T0*
_output_shapes	
:?2
	Reshape_6j
	Reshape_7Reshapesplit_2:output:0Const_3:output:0*
T0*
_output_shapes
:2
	Reshape_7j
	Reshape_8Reshapesplit_2:output:1Const_3:output:0*
T0*
_output_shapes
:2
	Reshape_8j
	Reshape_9Reshapesplit_2:output:2Const_3:output:0*
T0*
_output_shapes
:2
	Reshape_9l

Reshape_10Reshapesplit_2:output:3Const_3:output:0*
T0*
_output_shapes
:2

Reshape_10l

Reshape_11Reshapesplit_2:output:4Const_3:output:0*
T0*
_output_shapes
:2

Reshape_11l

Reshape_12Reshapesplit_2:output:5Const_3:output:0*
T0*
_output_shapes
:2

Reshape_12\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat/axis?
concatConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
N*
T0*
_output_shapes	
:?2
concati
CudnnRNN/input_cConst*
_output_shapes
: *
dtype0*
valueB
 *    2
CudnnRNN/input_c?
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat:output:0*
T0*Q
_output_shapes?
=:??????????????????:?????????: :*
rnn_modegru2

CudnnRNN}
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slicey
transpose_7/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_7/perm?
transpose_7	TransposeCudnnRNN:output:0transpose_7/perm:output:0*
T0*4
_output_shapes"
 :??????????????????2
transpose_7{
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:?????????*
squeeze_dims
 2	
Squeezef
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @2	
runtimej
IdentityIdentitystrided_slice:output:0*
T0*'
_output_shapes
:?????????2

Identityt

Identity_1Identitytranspose_7:y:0*
T0*4
_output_shapes"
 :??????????????????2

Identity_1h

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:?????????2

Identity_2W

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: 2

Identity_3"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*d
_input_shapesS
Q:??????????????????:?????????:Z:Z:Z*<
api_implements*(gru_d320c0db-d27b-490e-8348-8d4a20a9c88e*
api_preferred_deviceGPU*
go_backwards( *

time_major( :\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinit_h:FB

_output_shapes

:Z
 
_user_specified_namekernel:PL

_output_shapes

:Z
*
_user_specified_namerecurrent_kernel:D@

_output_shapes

:Z

_user_specified_namebias
?1
?
 __inference__wrapped_model_14294
	gru_input1
-sequential_1_gru_read_readvariableop_resource3
/sequential_1_gru_read_1_readvariableop_resource3
/sequential_1_gru_read_2_readvariableop_resource:
6sequential_1_dense_2_mlcmatmul_readvariableop_resource8
4sequential_1_dense_2_biasadd_readvariableop_resource
identity??+sequential_1/dense_2/BiasAdd/ReadVariableOp?-sequential_1/dense_2/MLCMatMul/ReadVariableOp?$sequential_1/gru/Read/ReadVariableOp?&sequential_1/gru/Read_1/ReadVariableOp?&sequential_1/gru/Read_2/ReadVariableOpi
sequential_1/gru/ShapeShape	gru_input*
T0*
_output_shapes
:2
sequential_1/gru/Shape?
$sequential_1/gru/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$sequential_1/gru/strided_slice/stack?
&sequential_1/gru/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&sequential_1/gru/strided_slice/stack_1?
&sequential_1/gru/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&sequential_1/gru/strided_slice/stack_2?
sequential_1/gru/strided_sliceStridedSlicesequential_1/gru/Shape:output:0-sequential_1/gru/strided_slice/stack:output:0/sequential_1/gru/strided_slice/stack_1:output:0/sequential_1/gru/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
sequential_1/gru/strided_slice~
sequential_1/gru/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
sequential_1/gru/zeros/mul/y?
sequential_1/gru/zeros/mulMul'sequential_1/gru/strided_slice:output:0%sequential_1/gru/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
sequential_1/gru/zeros/mul?
sequential_1/gru/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
sequential_1/gru/zeros/Less/y?
sequential_1/gru/zeros/LessLesssequential_1/gru/zeros/mul:z:0&sequential_1/gru/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
sequential_1/gru/zeros/Less?
sequential_1/gru/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2!
sequential_1/gru/zeros/packed/1?
sequential_1/gru/zeros/packedPack'sequential_1/gru/strided_slice:output:0(sequential_1/gru/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
sequential_1/gru/zeros/packed?
sequential_1/gru/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
sequential_1/gru/zeros/Const?
sequential_1/gru/zerosFill&sequential_1/gru/zeros/packed:output:0%sequential_1/gru/zeros/Const:output:0*
T0*'
_output_shapes
:?????????2
sequential_1/gru/zeros?
$sequential_1/gru/Read/ReadVariableOpReadVariableOp-sequential_1_gru_read_readvariableop_resource*
_output_shapes

:Z*
dtype02&
$sequential_1/gru/Read/ReadVariableOp?
sequential_1/gru/IdentityIdentity,sequential_1/gru/Read/ReadVariableOp:value:0*
T0*
_output_shapes

:Z2
sequential_1/gru/Identity?
&sequential_1/gru/Read_1/ReadVariableOpReadVariableOp/sequential_1_gru_read_1_readvariableop_resource*
_output_shapes

:Z*
dtype02(
&sequential_1/gru/Read_1/ReadVariableOp?
sequential_1/gru/Identity_1Identity.sequential_1/gru/Read_1/ReadVariableOp:value:0*
T0*
_output_shapes

:Z2
sequential_1/gru/Identity_1?
&sequential_1/gru/Read_2/ReadVariableOpReadVariableOp/sequential_1_gru_read_2_readvariableop_resource*
_output_shapes

:Z*
dtype02(
&sequential_1/gru/Read_2/ReadVariableOp?
sequential_1/gru/Identity_2Identity.sequential_1/gru/Read_2/ReadVariableOp:value:0*
T0*
_output_shapes

:Z2
sequential_1/gru/Identity_2?
 sequential_1/gru/PartitionedCallPartitionedCall	gru_inputsequential_1/gru/zeros:output:0"sequential_1/gru/Identity:output:0$sequential_1/gru/Identity_1:output:0$sequential_1/gru/Identity_2:output:0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *T
_output_shapesB
@:?????????:??????????:?????????: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *'
f"R 
__inference_standard_gru_139872"
 sequential_1/gru/PartitionedCall?
-sequential_1/dense_2/MLCMatMul/ReadVariableOpReadVariableOp6sequential_1_dense_2_mlcmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02/
-sequential_1/dense_2/MLCMatMul/ReadVariableOp?
sequential_1/dense_2/MLCMatMul	MLCMatMul)sequential_1/gru/PartitionedCall:output:05sequential_1/dense_2/MLCMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2 
sequential_1/dense_2/MLCMatMul?
+sequential_1/dense_2/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02-
+sequential_1/dense_2/BiasAdd/ReadVariableOp?
sequential_1/dense_2/BiasAddBiasAdd(sequential_1/dense_2/MLCMatMul:product:03sequential_1/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential_1/dense_2/BiasAdd?
sequential_1/dense_2/TanhTanh%sequential_1/dense_2/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
sequential_1/dense_2/Tanh?
IdentityIdentitysequential_1/dense_2/Tanh:y:0,^sequential_1/dense_2/BiasAdd/ReadVariableOp.^sequential_1/dense_2/MLCMatMul/ReadVariableOp%^sequential_1/gru/Read/ReadVariableOp'^sequential_1/gru/Read_1/ReadVariableOp'^sequential_1/gru/Read_2/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????:::::2Z
+sequential_1/dense_2/BiasAdd/ReadVariableOp+sequential_1/dense_2/BiasAdd/ReadVariableOp2^
-sequential_1/dense_2/MLCMatMul/ReadVariableOp-sequential_1/dense_2/MLCMatMul/ReadVariableOp2L
$sequential_1/gru/Read/ReadVariableOp$sequential_1/gru/Read/ReadVariableOp2P
&sequential_1/gru/Read_1/ReadVariableOp&sequential_1/gru/Read_1/ReadVariableOp2P
&sequential_1/gru/Read_2/ReadVariableOp&sequential_1/gru/Read_2/ReadVariableOp:W S
,
_output_shapes
:??????????
#
_user_specified_name	gru_input
??
?
8__inference___backward_gpu_gru_with_fallback_18022_18240
placeholder
placeholder_1
placeholder_2
placeholder_3/
+gradients_strided_slice_grad_shape_cudnnrnnB
>gradients_strided_slice_grad_mlcstridedslicegrad_strided_sliceA
=gradients_transpose_7_grad_invertpermutation_transpose_7_perm;
7gradients_transpose_7_grad_mlctransposegrad_transpose_7)
%gradients_squeeze_grad_shape_cudnnrnn!
gradients_zeros_like_cudnnrnn#
gradients_zeros_like_1_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims=
9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c3
/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h)
%gradients_concat_grad_add_concat_axis5
1gradients_reshape_1_grad_mlcreshapegrad_reshape_15
1gradients_reshape_2_grad_mlcreshapegrad_reshape_25
1gradients_reshape_3_grad_mlcreshapegrad_reshape_35
1gradients_reshape_4_grad_mlcreshapegrad_reshape_45
1gradients_reshape_5_grad_mlcreshapegrad_reshape_55
1gradients_reshape_6_grad_mlcreshapegrad_reshape_65
1gradients_reshape_7_grad_mlcreshapegrad_reshape_75
1gradients_reshape_8_grad_mlcreshapegrad_reshape_85
1gradients_reshape_9_grad_mlcreshapegrad_reshape_97
3gradients_reshape_10_grad_mlcreshapegrad_reshape_107
3gradients_reshape_11_grad_mlcreshapegrad_reshape_117
3gradients_reshape_12_grad_mlcreshapegrad_reshape_12A
=gradients_transpose_1_grad_invertpermutation_transpose_1_perm;
7gradients_transpose_1_grad_mlctransposegrad_transpose_1A
=gradients_transpose_2_grad_invertpermutation_transpose_2_perm;
7gradients_transpose_2_grad_mlctransposegrad_transpose_2A
=gradients_transpose_3_grad_invertpermutation_transpose_3_perm;
7gradients_transpose_3_grad_mlctransposegrad_transpose_3A
=gradients_transpose_4_grad_invertpermutation_transpose_4_perm;
7gradients_transpose_4_grad_mlctransposegrad_transpose_4A
=gradients_transpose_5_grad_invertpermutation_transpose_5_perm;
7gradients_transpose_5_grad_mlctransposegrad_transpose_5A
=gradients_transpose_6_grad_invertpermutation_transpose_6_perm;
7gradients_transpose_6_grad_mlctransposegrad_transpose_63
/gradients_split_2_grad_concat_split_2_split_dim5
1gradients_split_grad_mlcsplitgrad_split_split_dim+
'gradients_split_grad_mlcsplitgrad_split9
5gradients_split_1_grad_mlcsplitgrad_split_1_split_dim/
+gradients_split_1_grad_mlcsplitgrad_split_11
-gradients_reshape_grad_mlcreshapegrad_reshape
identity

identity_1

identity_2

identity_3

identity_4?u
gradients/grad_ys_0Identityplaceholder*
T0*'
_output_shapes
:?????????2
gradients/grad_ys_0|
gradients/grad_ys_1Identityplaceholder_1*
T0*,
_output_shapes
:??????????2
gradients/grad_ys_1w
gradients/grad_ys_2Identityplaceholder_2*
T0*'
_output_shapes
:?????????2
gradients/grad_ys_2f
gradients/grad_ys_3Identityplaceholder_3*
T0*
_output_shapes
: 2
gradients/grad_ys_3?
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:2$
"gradients/strided_slice_grad/Shape?
6gradients/strided_slice_grad/MLCStridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*
valueB:
?????????28
6gradients/strided_slice_grad/MLCStridedSliceGrad/begin?
4gradients/strided_slice_grad/MLCStridedSliceGrad/endConst*
_output_shapes
:*
dtype0*
valueB: 26
4gradients/strided_slice_grad/MLCStridedSliceGrad/end?
8gradients/strided_slice_grad/MLCStridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*
valueB:2:
8gradients/strided_slice_grad/MLCStridedSliceGrad/strides?
0gradients/strided_slice_grad/MLCStridedSliceGradMLCStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0?gradients/strided_slice_grad/MLCStridedSliceGrad/begin:output:0=gradients/strided_slice_grad/MLCStridedSliceGrad/end:output:0Agradients/strided_slice_grad/MLCStridedSliceGrad/strides:output:0gradients/grad_ys_0:output:0>gradients_strided_slice_grad_mlcstridedslicegrad_strided_slice*
Index0*
T0*,
_output_shapes
:??????????*
shrink_axis_mask22
0gradients/strided_slice_grad/MLCStridedSliceGrad?
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:2.
,gradients/transpose_7_grad/InvertPermutation?
+gradients/transpose_7_grad/MLCTransposeGradMLCTransposeGradgradients/grad_ys_1:output:00gradients/transpose_7_grad/InvertPermutation:y:07gradients_transpose_7_grad_mlctransposegrad_transpose_7*
T0*,
_output_shapes
:??????????2-
+gradients/transpose_7_grad/MLCTransposeGrad?
gradients/Squeeze_grad/ShapeShape%gradients_squeeze_grad_shape_cudnnrnn*
T0*
_output_shapes
:2
gradients/Squeeze_grad/Shape?
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*+
_output_shapes
:?????????2 
gradients/Squeeze_grad/Reshape?
gradients/AddNAddN9gradients/strided_slice_grad/MLCStridedSliceGrad:output:0/gradients/transpose_7_grad/MLCTransposeGrad:y:0*
N*
T0*C
_class9
75loc:@gradients/strided_slice_grad/MLCStridedSliceGrad*,
_output_shapes
:??????????2
gradients/AddNy
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
: 2
gradients/zeros_like?
gradients/zeros_like_1	ZerosLikegradients_zeros_like_1_cudnnrnn*
T0*
_output_shapes
:2
gradients/zeros_like_1?
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat+gradients_strided_slice_grad_shape_cudnnrnn%gradients_squeeze_grad_shape_cudnnrnngradients_zeros_like_cudnnrnngradients/AddN:sum:0'gradients/Squeeze_grad/Reshape:output:0gradients/zeros_like:y:0gradients_zeros_like_1_cudnnrnn*
T0*L
_output_shapes:
8:??????????:?????????: :?*
rnn_modegru2*
(gradients/CudnnRNN_grad/CudnnRNNBackprop?
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:2,
*gradients/transpose_grad/InvertPermutation?
)gradients/transpose_grad/MLCTransposeGradMLCTransposeGrad9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:02gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose*
T0*,
_output_shapes
:??????????2+
)gradients/transpose_grad/MLCTransposeGrad?
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
:2!
gradients/ExpandDims_grad/Shape?
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*'
_output_shapes
:?????????2#
!gradients/ExpandDims_grad/Reshape?
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:?2
gradients/concat_grad/Shape|
gradients/concat_grad/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_grad/add/y?
gradients/concat_grad/addAddV2%gradients_concat_grad_add_concat_axis$gradients/concat_grad/add/y:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/add?
)gradients/concat_grad/strided_slice/stackPack%gradients_concat_grad_add_concat_axis*
N*
T0*
_output_shapes
:2+
)gradients/concat_grad/strided_slice/stack?
+gradients/concat_grad/strided_slice/stack_1Packgradients/concat_grad/add:z:0*
N*
T0*
_output_shapes
:2-
+gradients/concat_grad/strided_slice/stack_1?
+gradients/concat_grad/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+gradients/concat_grad/strided_slice/stack_2?
#gradients/concat_grad/strided_sliceStridedSlice$gradients/concat_grad/Shape:output:02gradients/concat_grad/strided_slice/stack:output:04gradients/concat_grad/strided_slice/stack_1:output:04gradients/concat_grad/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#gradients/concat_grad/strided_slice?
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:?2
gradients/concat_grad/Shape_1?
gradients/concat_grad/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_grad/add_1/y?
gradients/concat_grad/add_1AddV2%gradients_concat_grad_add_concat_axis&gradients/concat_grad/add_1/y:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/add_1?
+gradients/concat_grad/strided_slice_1/stackPack%gradients_concat_grad_add_concat_axis*
N*
T0*
_output_shapes
:2-
+gradients/concat_grad/strided_slice_1/stack?
-gradients/concat_grad/strided_slice_1/stack_1Packgradients/concat_grad/add_1:z:0*
N*
T0*
_output_shapes
:2/
-gradients/concat_grad/strided_slice_1/stack_1?
-gradients/concat_grad/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-gradients/concat_grad/strided_slice_1/stack_2?
%gradients/concat_grad/strided_slice_1StridedSlice&gradients/concat_grad/Shape_1:output:04gradients/concat_grad/strided_slice_1/stack:output:06gradients/concat_grad/strided_slice_1/stack_1:output:06gradients/concat_grad/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%gradients/concat_grad/strided_slice_1?
gradients/concat_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:?2
gradients/concat_grad/Shape_2?
gradients/concat_grad/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_grad/add_2/y?
gradients/concat_grad/add_2AddV2%gradients_concat_grad_add_concat_axis&gradients/concat_grad/add_2/y:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/add_2?
+gradients/concat_grad/strided_slice_2/stackPack%gradients_concat_grad_add_concat_axis*
N*
T0*
_output_shapes
:2-
+gradients/concat_grad/strided_slice_2/stack?
-gradients/concat_grad/strided_slice_2/stack_1Packgradients/concat_grad/add_2:z:0*
N*
T0*
_output_shapes
:2/
-gradients/concat_grad/strided_slice_2/stack_1?
-gradients/concat_grad/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-gradients/concat_grad/strided_slice_2/stack_2?
%gradients/concat_grad/strided_slice_2StridedSlice&gradients/concat_grad/Shape_2:output:04gradients/concat_grad/strided_slice_2/stack:output:06gradients/concat_grad/strided_slice_2/stack_1:output:06gradients/concat_grad/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%gradients/concat_grad/strided_slice_2?
gradients/concat_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:?2
gradients/concat_grad/Shape_3?
gradients/concat_grad/add_3/yConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_grad/add_3/y?
gradients/concat_grad/add_3AddV2%gradients_concat_grad_add_concat_axis&gradients/concat_grad/add_3/y:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/add_3?
+gradients/concat_grad/strided_slice_3/stackPack%gradients_concat_grad_add_concat_axis*
N*
T0*
_output_shapes
:2-
+gradients/concat_grad/strided_slice_3/stack?
-gradients/concat_grad/strided_slice_3/stack_1Packgradients/concat_grad/add_3:z:0*
N*
T0*
_output_shapes
:2/
-gradients/concat_grad/strided_slice_3/stack_1?
-gradients/concat_grad/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-gradients/concat_grad/strided_slice_3/stack_2?
%gradients/concat_grad/strided_slice_3StridedSlice&gradients/concat_grad/Shape_3:output:04gradients/concat_grad/strided_slice_3/stack:output:06gradients/concat_grad/strided_slice_3/stack_1:output:06gradients/concat_grad/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%gradients/concat_grad/strided_slice_3?
gradients/concat_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:?2
gradients/concat_grad/Shape_4?
gradients/concat_grad/add_4/yConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_grad/add_4/y?
gradients/concat_grad/add_4AddV2%gradients_concat_grad_add_concat_axis&gradients/concat_grad/add_4/y:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/add_4?
+gradients/concat_grad/strided_slice_4/stackPack%gradients_concat_grad_add_concat_axis*
N*
T0*
_output_shapes
:2-
+gradients/concat_grad/strided_slice_4/stack?
-gradients/concat_grad/strided_slice_4/stack_1Packgradients/concat_grad/add_4:z:0*
N*
T0*
_output_shapes
:2/
-gradients/concat_grad/strided_slice_4/stack_1?
-gradients/concat_grad/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-gradients/concat_grad/strided_slice_4/stack_2?
%gradients/concat_grad/strided_slice_4StridedSlice&gradients/concat_grad/Shape_4:output:04gradients/concat_grad/strided_slice_4/stack:output:06gradients/concat_grad/strided_slice_4/stack_1:output:06gradients/concat_grad/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%gradients/concat_grad/strided_slice_4?
gradients/concat_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:?2
gradients/concat_grad/Shape_5?
gradients/concat_grad/add_5/yConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_grad/add_5/y?
gradients/concat_grad/add_5AddV2%gradients_concat_grad_add_concat_axis&gradients/concat_grad/add_5/y:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/add_5?
+gradients/concat_grad/strided_slice_5/stackPack%gradients_concat_grad_add_concat_axis*
N*
T0*
_output_shapes
:2-
+gradients/concat_grad/strided_slice_5/stack?
-gradients/concat_grad/strided_slice_5/stack_1Packgradients/concat_grad/add_5:z:0*
N*
T0*
_output_shapes
:2/
-gradients/concat_grad/strided_slice_5/stack_1?
-gradients/concat_grad/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-gradients/concat_grad/strided_slice_5/stack_2?
%gradients/concat_grad/strided_slice_5StridedSlice&gradients/concat_grad/Shape_5:output:04gradients/concat_grad/strided_slice_5/stack:output:06gradients/concat_grad/strided_slice_5/stack_1:output:06gradients/concat_grad/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%gradients/concat_grad/strided_slice_5?
gradients/concat_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape_6?
gradients/concat_grad/add_6/yConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_grad/add_6/y?
gradients/concat_grad/add_6AddV2%gradients_concat_grad_add_concat_axis&gradients/concat_grad/add_6/y:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/add_6?
+gradients/concat_grad/strided_slice_6/stackPack%gradients_concat_grad_add_concat_axis*
N*
T0*
_output_shapes
:2-
+gradients/concat_grad/strided_slice_6/stack?
-gradients/concat_grad/strided_slice_6/stack_1Packgradients/concat_grad/add_6:z:0*
N*
T0*
_output_shapes
:2/
-gradients/concat_grad/strided_slice_6/stack_1?
-gradients/concat_grad/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-gradients/concat_grad/strided_slice_6/stack_2?
%gradients/concat_grad/strided_slice_6StridedSlice&gradients/concat_grad/Shape_6:output:04gradients/concat_grad/strided_slice_6/stack:output:06gradients/concat_grad/strided_slice_6/stack_1:output:06gradients/concat_grad/strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%gradients/concat_grad/strided_slice_6?
gradients/concat_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape_7?
gradients/concat_grad/add_7/yConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_grad/add_7/y?
gradients/concat_grad/add_7AddV2%gradients_concat_grad_add_concat_axis&gradients/concat_grad/add_7/y:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/add_7?
+gradients/concat_grad/strided_slice_7/stackPack%gradients_concat_grad_add_concat_axis*
N*
T0*
_output_shapes
:2-
+gradients/concat_grad/strided_slice_7/stack?
-gradients/concat_grad/strided_slice_7/stack_1Packgradients/concat_grad/add_7:z:0*
N*
T0*
_output_shapes
:2/
-gradients/concat_grad/strided_slice_7/stack_1?
-gradients/concat_grad/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-gradients/concat_grad/strided_slice_7/stack_2?
%gradients/concat_grad/strided_slice_7StridedSlice&gradients/concat_grad/Shape_7:output:04gradients/concat_grad/strided_slice_7/stack:output:06gradients/concat_grad/strided_slice_7/stack_1:output:06gradients/concat_grad/strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%gradients/concat_grad/strided_slice_7?
gradients/concat_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape_8?
gradients/concat_grad/add_8/yConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_grad/add_8/y?
gradients/concat_grad/add_8AddV2%gradients_concat_grad_add_concat_axis&gradients/concat_grad/add_8/y:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/add_8?
+gradients/concat_grad/strided_slice_8/stackPack%gradients_concat_grad_add_concat_axis*
N*
T0*
_output_shapes
:2-
+gradients/concat_grad/strided_slice_8/stack?
-gradients/concat_grad/strided_slice_8/stack_1Packgradients/concat_grad/add_8:z:0*
N*
T0*
_output_shapes
:2/
-gradients/concat_grad/strided_slice_8/stack_1?
-gradients/concat_grad/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-gradients/concat_grad/strided_slice_8/stack_2?
%gradients/concat_grad/strided_slice_8StridedSlice&gradients/concat_grad/Shape_8:output:04gradients/concat_grad/strided_slice_8/stack:output:06gradients/concat_grad/strided_slice_8/stack_1:output:06gradients/concat_grad/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%gradients/concat_grad/strided_slice_8?
gradients/concat_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape_9?
gradients/concat_grad/add_9/yConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_grad/add_9/y?
gradients/concat_grad/add_9AddV2%gradients_concat_grad_add_concat_axis&gradients/concat_grad/add_9/y:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/add_9?
+gradients/concat_grad/strided_slice_9/stackPack%gradients_concat_grad_add_concat_axis*
N*
T0*
_output_shapes
:2-
+gradients/concat_grad/strided_slice_9/stack?
-gradients/concat_grad/strided_slice_9/stack_1Packgradients/concat_grad/add_9:z:0*
N*
T0*
_output_shapes
:2/
-gradients/concat_grad/strided_slice_9/stack_1?
-gradients/concat_grad/strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-gradients/concat_grad/strided_slice_9/stack_2?
%gradients/concat_grad/strided_slice_9StridedSlice&gradients/concat_grad/Shape_9:output:04gradients/concat_grad/strided_slice_9/stack:output:06gradients/concat_grad/strided_slice_9/stack_1:output:06gradients/concat_grad/strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%gradients/concat_grad/strided_slice_9?
gradients/concat_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB:2 
gradients/concat_grad/Shape_10?
gradients/concat_grad/add_10/yConst*
_output_shapes
: *
dtype0*
value	B :2 
gradients/concat_grad/add_10/y?
gradients/concat_grad/add_10AddV2%gradients_concat_grad_add_concat_axis'gradients/concat_grad/add_10/y:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/add_10?
,gradients/concat_grad/strided_slice_10/stackPack%gradients_concat_grad_add_concat_axis*
N*
T0*
_output_shapes
:2.
,gradients/concat_grad/strided_slice_10/stack?
.gradients/concat_grad/strided_slice_10/stack_1Pack gradients/concat_grad/add_10:z:0*
N*
T0*
_output_shapes
:20
.gradients/concat_grad/strided_slice_10/stack_1?
.gradients/concat_grad/strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.gradients/concat_grad/strided_slice_10/stack_2?
&gradients/concat_grad/strided_slice_10StridedSlice'gradients/concat_grad/Shape_10:output:05gradients/concat_grad/strided_slice_10/stack:output:07gradients/concat_grad/strided_slice_10/stack_1:output:07gradients/concat_grad/strided_slice_10/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2(
&gradients/concat_grad/strided_slice_10?
gradients/concat_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB:2 
gradients/concat_grad/Shape_11?
gradients/concat_grad/add_11/yConst*
_output_shapes
: *
dtype0*
value	B :2 
gradients/concat_grad/add_11/y?
gradients/concat_grad/add_11AddV2%gradients_concat_grad_add_concat_axis'gradients/concat_grad/add_11/y:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/add_11?
,gradients/concat_grad/strided_slice_11/stackPack%gradients_concat_grad_add_concat_axis*
N*
T0*
_output_shapes
:2.
,gradients/concat_grad/strided_slice_11/stack?
.gradients/concat_grad/strided_slice_11/stack_1Pack gradients/concat_grad/add_11:z:0*
N*
T0*
_output_shapes
:20
.gradients/concat_grad/strided_slice_11/stack_1?
.gradients/concat_grad/strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.gradients/concat_grad/strided_slice_11/stack_2?
&gradients/concat_grad/strided_slice_11StridedSlice'gradients/concat_grad/Shape_11:output:05gradients/concat_grad/strided_slice_11/stack:output:07gradients/concat_grad/strided_slice_11/stack_1:output:07gradients/concat_grad/strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2(
&gradients/concat_grad/strided_slice_11?
gradients/concat_grad/packedPack,gradients/concat_grad/strided_slice:output:0.gradients/concat_grad/strided_slice_1:output:0.gradients/concat_grad/strided_slice_2:output:0.gradients/concat_grad/strided_slice_3:output:0.gradients/concat_grad/strided_slice_4:output:0.gradients/concat_grad/strided_slice_5:output:0.gradients/concat_grad/strided_slice_6:output:0.gradients/concat_grad/strided_slice_7:output:0.gradients/concat_grad/strided_slice_8:output:0.gradients/concat_grad/strided_slice_9:output:0/gradients/concat_grad/strided_slice_10:output:0/gradients/concat_grad/strided_slice_11:output:0*
N*
T0*
_output_shapes
:2
gradients/concat_grad/packed?
%gradients/concat_grad/MLCConcatV2GradMLCConcatV2Grad:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0%gradients/concat_grad/packed:output:0%gradients_concat_grad_add_concat_axis$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0&gradients/concat_grad/Shape_2:output:0&gradients/concat_grad/Shape_3:output:0&gradients/concat_grad/Shape_4:output:0&gradients/concat_grad/Shape_5:output:0&gradients/concat_grad/Shape_6:output:0&gradients/concat_grad/Shape_7:output:0&gradients/concat_grad/Shape_8:output:0&gradients/concat_grad/Shape_9:output:0'gradients/concat_grad/Shape_10:output:0'gradients/concat_grad/Shape_11:output:0/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat*
N*
T0*

Tlen0*b
_output_shapesP
N:?:?:?:?:?:?::::::2'
%gradients/concat_grad/MLCConcatV2Grad?
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
gradients/Reshape_1_grad/Shape?
'gradients/Reshape_1_grad/MLCReshapeGradMLCReshapeGrad.gradients/concat_grad/MLCConcatV2Grad:output:0'gradients/Reshape_1_grad/Shape:output:01gradients_reshape_1_grad_mlcreshapegrad_reshape_1*
T0*
_output_shapes

:2)
'gradients/Reshape_1_grad/MLCReshapeGrad?
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
gradients/Reshape_2_grad/Shape?
'gradients/Reshape_2_grad/MLCReshapeGradMLCReshapeGrad.gradients/concat_grad/MLCConcatV2Grad:output:1'gradients/Reshape_2_grad/Shape:output:01gradients_reshape_2_grad_mlcreshapegrad_reshape_2*
T0*
_output_shapes

:2)
'gradients/Reshape_2_grad/MLCReshapeGrad?
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
gradients/Reshape_3_grad/Shape?
'gradients/Reshape_3_grad/MLCReshapeGradMLCReshapeGrad.gradients/concat_grad/MLCConcatV2Grad:output:2'gradients/Reshape_3_grad/Shape:output:01gradients_reshape_3_grad_mlcreshapegrad_reshape_3*
T0*
_output_shapes

:2)
'gradients/Reshape_3_grad/MLCReshapeGrad?
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
gradients/Reshape_4_grad/Shape?
'gradients/Reshape_4_grad/MLCReshapeGradMLCReshapeGrad.gradients/concat_grad/MLCConcatV2Grad:output:3'gradients/Reshape_4_grad/Shape:output:01gradients_reshape_4_grad_mlcreshapegrad_reshape_4*
T0*
_output_shapes

:2)
'gradients/Reshape_4_grad/MLCReshapeGrad?
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
gradients/Reshape_5_grad/Shape?
'gradients/Reshape_5_grad/MLCReshapeGradMLCReshapeGrad.gradients/concat_grad/MLCConcatV2Grad:output:4'gradients/Reshape_5_grad/Shape:output:01gradients_reshape_5_grad_mlcreshapegrad_reshape_5*
T0*
_output_shapes

:2)
'gradients/Reshape_5_grad/MLCReshapeGrad?
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
gradients/Reshape_6_grad/Shape?
'gradients/Reshape_6_grad/MLCReshapeGradMLCReshapeGrad.gradients/concat_grad/MLCConcatV2Grad:output:5'gradients/Reshape_6_grad/Shape:output:01gradients_reshape_6_grad_mlcreshapegrad_reshape_6*
T0*
_output_shapes

:2)
'gradients/Reshape_6_grad/MLCReshapeGrad?
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2 
gradients/Reshape_7_grad/Shape?
'gradients/Reshape_7_grad/MLCReshapeGradMLCReshapeGrad.gradients/concat_grad/MLCConcatV2Grad:output:6'gradients/Reshape_7_grad/Shape:output:01gradients_reshape_7_grad_mlcreshapegrad_reshape_7*
T0*
_output_shapes
:2)
'gradients/Reshape_7_grad/MLCReshapeGrad?
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2 
gradients/Reshape_8_grad/Shape?
'gradients/Reshape_8_grad/MLCReshapeGradMLCReshapeGrad.gradients/concat_grad/MLCConcatV2Grad:output:7'gradients/Reshape_8_grad/Shape:output:01gradients_reshape_8_grad_mlcreshapegrad_reshape_8*
T0*
_output_shapes
:2)
'gradients/Reshape_8_grad/MLCReshapeGrad?
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2 
gradients/Reshape_9_grad/Shape?
'gradients/Reshape_9_grad/MLCReshapeGradMLCReshapeGrad.gradients/concat_grad/MLCConcatV2Grad:output:8'gradients/Reshape_9_grad/Shape:output:01gradients_reshape_9_grad_mlcreshapegrad_reshape_9*
T0*
_output_shapes
:2)
'gradients/Reshape_9_grad/MLCReshapeGrad?
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2!
gradients/Reshape_10_grad/Shape?
(gradients/Reshape_10_grad/MLCReshapeGradMLCReshapeGrad.gradients/concat_grad/MLCConcatV2Grad:output:9(gradients/Reshape_10_grad/Shape:output:03gradients_reshape_10_grad_mlcreshapegrad_reshape_10*
T0*
_output_shapes
:2*
(gradients/Reshape_10_grad/MLCReshapeGrad?
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2!
gradients/Reshape_11_grad/Shape?
(gradients/Reshape_11_grad/MLCReshapeGradMLCReshapeGrad/gradients/concat_grad/MLCConcatV2Grad:output:10(gradients/Reshape_11_grad/Shape:output:03gradients_reshape_11_grad_mlcreshapegrad_reshape_11*
T0*
_output_shapes
:2*
(gradients/Reshape_11_grad/MLCReshapeGrad?
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2!
gradients/Reshape_12_grad/Shape?
(gradients/Reshape_12_grad/MLCReshapeGradMLCReshapeGrad/gradients/concat_grad/MLCConcatV2Grad:output:11(gradients/Reshape_12_grad/Shape:output:03gradients_reshape_12_grad_mlcreshapegrad_reshape_12*
T0*
_output_shapes
:2*
(gradients/Reshape_12_grad/MLCReshapeGrad?
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:2.
,gradients/transpose_1_grad/InvertPermutation?
+gradients/transpose_1_grad/MLCTransposeGradMLCTransposeGrad0gradients/Reshape_1_grad/MLCReshapeGrad:output:00gradients/transpose_1_grad/InvertPermutation:y:07gradients_transpose_1_grad_mlctransposegrad_transpose_1*
T0*
_output_shapes

:2-
+gradients/transpose_1_grad/MLCTransposeGrad?
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:2.
,gradients/transpose_2_grad/InvertPermutation?
+gradients/transpose_2_grad/MLCTransposeGradMLCTransposeGrad0gradients/Reshape_2_grad/MLCReshapeGrad:output:00gradients/transpose_2_grad/InvertPermutation:y:07gradients_transpose_2_grad_mlctransposegrad_transpose_2*
T0*
_output_shapes

:2-
+gradients/transpose_2_grad/MLCTransposeGrad?
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:2.
,gradients/transpose_3_grad/InvertPermutation?
+gradients/transpose_3_grad/MLCTransposeGradMLCTransposeGrad0gradients/Reshape_3_grad/MLCReshapeGrad:output:00gradients/transpose_3_grad/InvertPermutation:y:07gradients_transpose_3_grad_mlctransposegrad_transpose_3*
T0*
_output_shapes

:2-
+gradients/transpose_3_grad/MLCTransposeGrad?
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:2.
,gradients/transpose_4_grad/InvertPermutation?
+gradients/transpose_4_grad/MLCTransposeGradMLCTransposeGrad0gradients/Reshape_4_grad/MLCReshapeGrad:output:00gradients/transpose_4_grad/InvertPermutation:y:07gradients_transpose_4_grad_mlctransposegrad_transpose_4*
T0*
_output_shapes

:2-
+gradients/transpose_4_grad/MLCTransposeGrad?
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:2.
,gradients/transpose_5_grad/InvertPermutation?
+gradients/transpose_5_grad/MLCTransposeGradMLCTransposeGrad0gradients/Reshape_5_grad/MLCReshapeGrad:output:00gradients/transpose_5_grad/InvertPermutation:y:07gradients_transpose_5_grad_mlctransposegrad_transpose_5*
T0*
_output_shapes

:2-
+gradients/transpose_5_grad/MLCTransposeGrad?
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:2.
,gradients/transpose_6_grad/InvertPermutation?
+gradients/transpose_6_grad/MLCTransposeGradMLCTransposeGrad0gradients/Reshape_6_grad/MLCReshapeGrad:output:00gradients/transpose_6_grad/InvertPermutation:y:07gradients_transpose_6_grad_mlctransposegrad_transpose_6*
T0*
_output_shapes

:2-
+gradients/transpose_6_grad/MLCTransposeGrad?
gradients/split_2_grad/concatConcatV20gradients/Reshape_7_grad/MLCReshapeGrad:output:00gradients/Reshape_8_grad/MLCReshapeGrad:output:00gradients/Reshape_9_grad/MLCReshapeGrad:output:01gradients/Reshape_10_grad/MLCReshapeGrad:output:01gradients/Reshape_11_grad/MLCReshapeGrad:output:01gradients/Reshape_12_grad/MLCReshapeGrad:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes	
:?2
gradients/split_2_grad/concat?
!gradients/split_grad/MLCSplitGradMLCSplitGrad/gradients/transpose_1_grad/MLCTransposeGrad:y:0/gradients/transpose_2_grad/MLCTransposeGrad:y:0/gradients/transpose_3_grad/MLCTransposeGrad:y:01gradients_split_grad_mlcsplitgrad_split_split_dim'gradients_split_grad_mlcsplitgrad_split*
N*
T0*
_output_shapes

:Z2#
!gradients/split_grad/MLCSplitGrad?
#gradients/split_1_grad/MLCSplitGradMLCSplitGrad/gradients/transpose_4_grad/MLCTransposeGrad:y:0/gradients/transpose_5_grad/MLCTransposeGrad:y:0/gradients/transpose_6_grad/MLCTransposeGrad:y:05gradients_split_1_grad_mlcsplitgrad_split_1_split_dim+gradients_split_1_grad_mlcsplitgrad_split_1*
N*
T0*
_output_shapes

:Z2%
#gradients/split_1_grad/MLCSplitGrad?
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"   Z   2
gradients/Reshape_grad/Shape?
%gradients/Reshape_grad/MLCReshapeGradMLCReshapeGrad&gradients/split_2_grad/concat:output:0%gradients/Reshape_grad/Shape:output:0-gradients_reshape_grad_mlcreshapegrad_reshape*
T0*
_output_shapes

:Z2'
%gradients/Reshape_grad/MLCReshapeGrad?
IdentityIdentity-gradients/transpose_grad/MLCTransposeGrad:y:0*
T0*,
_output_shapes
:??????????2

Identity?

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0*
T0*'
_output_shapes
:?????????2

Identity_1y

Identity_2Identity*gradients/split_grad/MLCSplitGrad:output:0*
T0*
_output_shapes

:Z2

Identity_2{

Identity_3Identity,gradients/split_1_grad/MLCSplitGrad:output:0*
T0*
_output_shapes

:Z2

Identity_3}

Identity_4Identity.gradients/Reshape_grad/MLCReshapeGrad:output:0*
T0*
_output_shapes

:Z2

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*?
_input_shapes?
?:?????????:??????????:?????????: :??????????:?????????::??????????:?????????: ::??????????:?????????: :?::?????????: :?:?:?:?:?:?::::::::::::::::::: : :: ::?*<
api_implements*(gru_2db1bf7b-7db5-4f71-98ab-d0201f7417d2*
api_preferred_deviceGPU*@
forward_function_name'%__forward_gpu_gru_with_fallback_18239*
go_backwards( *

time_major( :- )
'
_output_shapes
:?????????:2.
,
_output_shapes
:??????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :2.
,
_output_shapes
:??????????:-)
'
_output_shapes
:?????????: 

_output_shapes
::2.
,
_output_shapes
:??????????:1-
+
_output_shapes
:?????????:	

_output_shapes
: :


_output_shapes
::2.
,
_output_shapes
:??????????:1-
+
_output_shapes
:?????????:

_output_shapes
: :!

_output_shapes	
:?: 

_output_shapes
::-)
'
_output_shapes
:?????????:

_output_shapes
: :!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::$ 

_output_shapes

::  

_output_shapes
::$! 

_output_shapes

:: "

_output_shapes
::$# 

_output_shapes

:: $

_output_shapes
::$% 

_output_shapes

:: &

_output_shapes
::$' 

_output_shapes

:: (

_output_shapes
::$) 

_output_shapes

::*

_output_shapes
: :+

_output_shapes
: :$, 

_output_shapes

::-

_output_shapes
: :$. 

_output_shapes

::!/

_output_shapes	
:?
?E
?
__inference_standard_gru_18448

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3X
unstackUnpackbias*
T0* 
_output_shapes
:Z:Z*	
num2	
unstacku
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????2
	transposeK
ShapeShapetranspose:y:0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_1n
MatMulMatMulstrided_slice_1:output:0kernel*
T0*'
_output_shapes
:?????????Z2
MatMuls
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*'
_output_shapes
:?????????Z2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:?????????:?????????:?????????*
	num_split2
splitj
MatMul_1MatMulinit_hrecurrent_kernel*
T0*'
_output_shapes
:?????????Z2

MatMul_1y
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*'
_output_shapes
:?????????Z2
	BiasAdd_1T
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dim?
split_1Splitsplit_1/split_dim:output:0BiasAdd_1:output:0*
T0*M
_output_shapes;
9:?????????:?????????:?????????*
	num_split2	
split_1g
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:?????????2
addX
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:?????????2	
Sigmoidk
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:?????????2
add_1^
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:?????????2
	Sigmoid_1d
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:?????????2
mulb
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:?????????2
add_2Q
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:?????????2
Tanh\
mul_1MulSigmoid:y:0init_h*
T0*'
_output_shapes
:?????????2
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
sub/x`
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:?????????2
subZ
mul_2Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:?????????2
mul_2_
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:?????????2
add_3?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileStatelessWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelunstack:output:0recurrent_kernelunstack:output:1*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*S
_output_shapesA
?: : : : :?????????: : :Z:Z:Z:Z* 
_read_only_resource_inputs
 *
bodyR
while_body_18357*
condR
while_cond_18356*R
output_shapesA
?: : : : :?????????: : :Z:Z:Z:Z*
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_2y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  ??2	
runtimel
IdentityIdentitystrided_slice_2:output:0*
T0*'
_output_shapes
:?????????2

Identityt

Identity_1Identitytranspose_1:y:0*
T0*4
_output_shapes"
 :??????????????????2

Identity_1f

Identity_2Identitywhile:output:4*
T0*'
_output_shapes
:?????????2

Identity_2W

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: 2

Identity_3"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*d
_input_shapesS
Q:??????????????????:?????????:Z:Z:Z*<
api_implements*(gru_7fd55799-5a12-43e2-bc93-7077640e6183*
api_preferred_deviceCPU*
go_backwards( *

time_major( :\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinit_h:FB

_output_shapes

:Z
 
_user_specified_namekernel:PL

_output_shapes

:Z
*
_user_specified_namerecurrent_kernel:D@

_output_shapes

:Z

_user_specified_namebias
?E
?
__inference_standard_gru_19877

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3X
unstackUnpackbias*
T0* 
_output_shapes
:Z:Z*	
num2	
unstacku
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm{
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:??????????2
	transposeK
ShapeShapetranspose:y:0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_1n
MatMulMatMulstrided_slice_1:output:0kernel*
T0*'
_output_shapes
:?????????Z2
MatMuls
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*'
_output_shapes
:?????????Z2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:?????????:?????????:?????????*
	num_split2
splitj
MatMul_1MatMulinit_hrecurrent_kernel*
T0*'
_output_shapes
:?????????Z2

MatMul_1y
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*'
_output_shapes
:?????????Z2
	BiasAdd_1T
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dim?
split_1Splitsplit_1/split_dim:output:0BiasAdd_1:output:0*
T0*M
_output_shapes;
9:?????????:?????????:?????????*
	num_split2	
split_1g
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:?????????2
addX
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:?????????2	
Sigmoidk
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:?????????2
add_1^
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:?????????2
	Sigmoid_1d
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:?????????2
mulb
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:?????????2
add_2Q
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:?????????2
Tanh\
mul_1MulSigmoid:y:0init_h*
T0*'
_output_shapes
:?????????2
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
sub/x`
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:?????????2
subZ
mul_2Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:?????????2
mul_2_
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:?????????2
add_3?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileStatelessWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelunstack:output:0recurrent_kernelunstack:output:1*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*S
_output_shapesA
?: : : : :?????????: : :Z:Z:Z:Z* 
_read_only_resource_inputs
 *
bodyR
while_body_19786*
condR
while_cond_19785*R
output_shapesA
?: : : : :?????????: : :Z:Z:Z:Z*
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_2y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:??????????2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  ??2	
runtimel
IdentityIdentitystrided_slice_2:output:0*
T0*'
_output_shapes
:?????????2

Identityl

Identity_1Identitytranspose_1:y:0*
T0*,
_output_shapes
:??????????2

Identity_1f

Identity_2Identitywhile:output:4*
T0*'
_output_shapes
:?????????2

Identity_2W

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: 2

Identity_3"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*\
_input_shapesK
I:??????????:?????????:Z:Z:Z*<
api_implements*(gru_c4a462d1-fe50-45b7-baf6-20215ecbec52*
api_preferred_deviceCPU*
go_backwards( *

time_major( :T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinit_h:FB

_output_shapes

:Z
 
_user_specified_namekernel:PL

_output_shapes

:Z
*
_user_specified_namerecurrent_kernel:D@

_output_shapes

:Z

_user_specified_namebias
?<
?
'__inference_gpu_gru_with_fallback_19487

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3?u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm{
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:??????????2
	transposeb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims/dim}

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2

ExpandDimsP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
splitSplitsplit/split_dim:output:0kernel*
T0*2
_output_shapes 
:::*
	num_split2
splitT
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dim?
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*2
_output_shapes 
:::*
	num_split2	
split_1q
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
Reshape/shapea
ReshapeReshapebiasReshape/shape:output:0*
T0*
_output_shapes	
:?2	
ReshapeT
Const_2Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_2h
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_2/split_dim?
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*8
_output_shapes&
$::::::*
	num_split2	
split_2e
Const_3Const*
_output_shapes
:*
dtype0*
valueB:
?????????2	
Const_3u
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_1/perm{
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes

:2
transpose_1j
	Reshape_1Reshapetranspose_1:y:0Const_3:output:0*
T0*
_output_shapes	
:?2
	Reshape_1u
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_2/perm{
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes

:2
transpose_2j
	Reshape_2Reshapetranspose_2:y:0Const_3:output:0*
T0*
_output_shapes	
:?2
	Reshape_2u
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_3/perm{
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

:2
transpose_3j
	Reshape_3Reshapetranspose_3:y:0Const_3:output:0*
T0*
_output_shapes	
:?2
	Reshape_3u
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_4/perm}
transpose_4	Transposesplit_1:output:0transpose_4/perm:output:0*
T0*
_output_shapes

:2
transpose_4j
	Reshape_4Reshapetranspose_4:y:0Const_3:output:0*
T0*
_output_shapes	
:?2
	Reshape_4u
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_5/perm}
transpose_5	Transposesplit_1:output:1transpose_5/perm:output:0*
T0*
_output_shapes

:2
transpose_5j
	Reshape_5Reshapetranspose_5:y:0Const_3:output:0*
T0*
_output_shapes	
:?2
	Reshape_5u
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_6/perm}
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T0*
_output_shapes

:2
transpose_6j
	Reshape_6Reshapetranspose_6:y:0Const_3:output:0*
T0*
_output_shapes	
:?2
	Reshape_6j
	Reshape_7Reshapesplit_2:output:0Const_3:output:0*
T0*
_output_shapes
:2
	Reshape_7j
	Reshape_8Reshapesplit_2:output:1Const_3:output:0*
T0*
_output_shapes
:2
	Reshape_8j
	Reshape_9Reshapesplit_2:output:2Const_3:output:0*
T0*
_output_shapes
:2
	Reshape_9l

Reshape_10Reshapesplit_2:output:3Const_3:output:0*
T0*
_output_shapes
:2

Reshape_10l

Reshape_11Reshapesplit_2:output:4Const_3:output:0*
T0*
_output_shapes
:2

Reshape_11l

Reshape_12Reshapesplit_2:output:5Const_3:output:0*
T0*
_output_shapes
:2

Reshape_12\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat/axis?
concatConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
N*
T0*
_output_shapes	
:?2
concati
CudnnRNN/input_cConst*
_output_shapes
: *
dtype0*
valueB
 *    2
CudnnRNN/input_c?
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat:output:0*
T0*I
_output_shapes7
5:??????????:?????????: :*
rnn_modegru2

CudnnRNN}
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slicey
transpose_7/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_7/perm?
transpose_7	TransposeCudnnRNN:output:0transpose_7/perm:output:0*
T0*,
_output_shapes
:??????????2
transpose_7{
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:?????????*
squeeze_dims
 2	
Squeezef
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @2	
runtimej
IdentityIdentitystrided_slice:output:0*
T0*'
_output_shapes
:?????????2

Identityl

Identity_1Identitytranspose_7:y:0*
T0*,
_output_shapes
:??????????2

Identity_1h

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:?????????2

Identity_2W

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: 2

Identity_3"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*\
_input_shapesK
I:??????????:?????????:Z:Z:Z*<
api_implements*(gru_e9d87f5a-6bce-4ea5-b775-fb635bd39d20*
api_preferred_deviceGPU*
go_backwards( *

time_major( :T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinit_h:FB

_output_shapes

:Z
 
_user_specified_namekernel:PL

_output_shapes

:Z
*
_user_specified_namerecurrent_kernel:D@

_output_shapes

:Z

_user_specified_namebias
?<
?
'__inference_gpu_gru_with_fallback_16914

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3?u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm{
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:??????????2
	transposeb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims/dim}

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2

ExpandDimsP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
splitSplitsplit/split_dim:output:0kernel*
T0*2
_output_shapes 
:::*
	num_split2
splitT
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dim?
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*2
_output_shapes 
:::*
	num_split2	
split_1q
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
Reshape/shapea
ReshapeReshapebiasReshape/shape:output:0*
T0*
_output_shapes	
:?2	
ReshapeT
Const_2Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_2h
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_2/split_dim?
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*8
_output_shapes&
$::::::*
	num_split2	
split_2e
Const_3Const*
_output_shapes
:*
dtype0*
valueB:
?????????2	
Const_3u
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_1/perm{
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes

:2
transpose_1j
	Reshape_1Reshapetranspose_1:y:0Const_3:output:0*
T0*
_output_shapes	
:?2
	Reshape_1u
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_2/perm{
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes

:2
transpose_2j
	Reshape_2Reshapetranspose_2:y:0Const_3:output:0*
T0*
_output_shapes	
:?2
	Reshape_2u
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_3/perm{
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

:2
transpose_3j
	Reshape_3Reshapetranspose_3:y:0Const_3:output:0*
T0*
_output_shapes	
:?2
	Reshape_3u
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_4/perm}
transpose_4	Transposesplit_1:output:0transpose_4/perm:output:0*
T0*
_output_shapes

:2
transpose_4j
	Reshape_4Reshapetranspose_4:y:0Const_3:output:0*
T0*
_output_shapes	
:?2
	Reshape_4u
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_5/perm}
transpose_5	Transposesplit_1:output:1transpose_5/perm:output:0*
T0*
_output_shapes

:2
transpose_5j
	Reshape_5Reshapetranspose_5:y:0Const_3:output:0*
T0*
_output_shapes	
:?2
	Reshape_5u
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_6/perm}
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T0*
_output_shapes

:2
transpose_6j
	Reshape_6Reshapetranspose_6:y:0Const_3:output:0*
T0*
_output_shapes	
:?2
	Reshape_6j
	Reshape_7Reshapesplit_2:output:0Const_3:output:0*
T0*
_output_shapes
:2
	Reshape_7j
	Reshape_8Reshapesplit_2:output:1Const_3:output:0*
T0*
_output_shapes
:2
	Reshape_8j
	Reshape_9Reshapesplit_2:output:2Const_3:output:0*
T0*
_output_shapes
:2
	Reshape_9l

Reshape_10Reshapesplit_2:output:3Const_3:output:0*
T0*
_output_shapes
:2

Reshape_10l

Reshape_11Reshapesplit_2:output:4Const_3:output:0*
T0*
_output_shapes
:2

Reshape_11l

Reshape_12Reshapesplit_2:output:5Const_3:output:0*
T0*
_output_shapes
:2

Reshape_12\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat/axis?
concatConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
N*
T0*
_output_shapes	
:?2
concati
CudnnRNN/input_cConst*
_output_shapes
: *
dtype0*
valueB
 *    2
CudnnRNN/input_c?
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat:output:0*
T0*I
_output_shapes7
5:??????????:?????????: :*
rnn_modegru2

CudnnRNN}
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slicey
transpose_7/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_7/perm?
transpose_7	TransposeCudnnRNN:output:0transpose_7/perm:output:0*
T0*,
_output_shapes
:??????????2
transpose_7{
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:?????????*
squeeze_dims
 2	
Squeezef
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @2	
runtimej
IdentityIdentitystrided_slice:output:0*
T0*'
_output_shapes
:?????????2

Identityl

Identity_1Identitytranspose_7:y:0*
T0*,
_output_shapes
:??????????2

Identity_1h

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:?????????2

Identity_2W

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: 2

Identity_3"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*\
_input_shapesK
I:??????????:?????????:Z:Z:Z*<
api_implements*(gru_8d3b7182-ad7a-489b-86fa-c30b709435ce*
api_preferred_deviceGPU*
go_backwards( *

time_major( :T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinit_h:FB

_output_shapes

:Z
 
_user_specified_namekernel:PL

_output_shapes

:Z
*
_user_specified_namerecurrent_kernel:D@

_output_shapes

:Z

_user_specified_namebias
?M
?
%__forward_gpu_gru_with_fallback_19705

inputs
init_h_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3
cudnnrnn
strided_slice
transpose_7_perm
transpose_7

cudnnrnn_0

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
cudnnrnn_input_c

concat
transpose_perm

init_h
concat_axis
	reshape_1
	reshape_2
	reshape_3
	reshape_4
	reshape_5
	reshape_6
	reshape_7
	reshape_8
	reshape_9

reshape_10

reshape_11

reshape_12
transpose_1_perm
transpose_1
transpose_2_perm
transpose_2
transpose_3_perm
transpose_3
transpose_4_perm
transpose_4
transpose_5_perm
transpose_5
transpose_6_perm
transpose_6
split_2_split_dim
split_split_dim	
split
split_1_split_dim
split_1
reshape?u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permO
transpose_0	Transposeinputstranspose/perm:output:0*
T02
	transposeb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims/dim

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2

ExpandDimsP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dimU
split_0Splitsplit/split_dim:output:0kernel*
T0*
	num_split2
splitT
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dime
	split_1_0Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*
	num_split2	
split_1q
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
Reshape/shapea
ReshapeReshapebiasReshape/shape:output:0*
T0*
_output_shapes	
:?2	
ReshapeT
Const_2Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_2h
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_2/split_dim?
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*8
_output_shapes&
$::::::*
	num_split2	
split_2e
Const_3Const*
_output_shapes
:*
dtype0*
valueB:
?????????2	
Const_3u
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_1/perm_
transpose_1_0	Transposesplit_0:output:0transpose_1/perm:output:0*
T02
transpose_1l
	Reshape_1Reshapetranspose_1_0:y:0Const_3:output:0*
T0*
_output_shapes	
:?2
	Reshape_1u
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_2/perm_
transpose_2_0	Transposesplit_0:output:1transpose_2/perm:output:0*
T02
transpose_2l
	Reshape_2Reshapetranspose_2_0:y:0Const_3:output:0*
T0*
_output_shapes	
:?2
	Reshape_2u
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_3/perm_
transpose_3_0	Transposesplit_0:output:2transpose_3/perm:output:0*
T02
transpose_3l
	Reshape_3Reshapetranspose_3_0:y:0Const_3:output:0*
T0*
_output_shapes	
:?2
	Reshape_3u
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_4/perma
transpose_4_0	Transposesplit_1_0:output:0transpose_4/perm:output:0*
T02
transpose_4l
	Reshape_4Reshapetranspose_4_0:y:0Const_3:output:0*
T0*
_output_shapes	
:?2
	Reshape_4u
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_5/perma
transpose_5_0	Transposesplit_1_0:output:1transpose_5/perm:output:0*
T02
transpose_5l
	Reshape_5Reshapetranspose_5_0:y:0Const_3:output:0*
T0*
_output_shapes	
:?2
	Reshape_5u
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_6/perma
transpose_6_0	Transposesplit_1_0:output:2transpose_6/perm:output:0*
T02
transpose_6l
	Reshape_6Reshapetranspose_6_0:y:0Const_3:output:0*
T0*
_output_shapes	
:?2
	Reshape_6j
	Reshape_7Reshapesplit_2:output:0Const_3:output:0*
T0*
_output_shapes
:2
	Reshape_7j
	Reshape_8Reshapesplit_2:output:1Const_3:output:0*
T0*
_output_shapes
:2
	Reshape_8j
	Reshape_9Reshapesplit_2:output:2Const_3:output:0*
T0*
_output_shapes
:2
	Reshape_9l

Reshape_10Reshapesplit_2:output:3Const_3:output:0*
T0*
_output_shapes
:2

Reshape_10l

Reshape_11Reshapesplit_2:output:4Const_3:output:0*
T0*
_output_shapes
:2

Reshape_11l

Reshape_12Reshapesplit_2:output:5Const_3:output:0*
T0*
_output_shapes
:2

Reshape_12\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat/axis?
concat_0ConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
N*
T02
concati
CudnnRNN/input_cConst*
_output_shapes
: *
dtype0*
valueB
 *    2
CudnnRNN/input_c?
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat_0:output:0*
T0*I
_output_shapes7
5:??????????:?????????: :*
rnn_modegru2

CudnnRNN}
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_slice_0StridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
shrink_axis_mask2
strided_slicey
transpose_7/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_7/perm`
transpose_7_0	TransposeCudnnRNN:output:0transpose_7/perm:output:0*
T02
transpose_7{
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:?????????*
squeeze_dims
 2	
Squeezef
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @2	
runtimel
IdentityIdentitystrided_slice_0:output:0*
T0*'
_output_shapes
:?????????2

Identityn

Identity_1Identitytranspose_7_0:y:0*
T0*,
_output_shapes
:??????????2

Identity_1h

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:?????????2

Identity_2W

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: 2

Identity_3"
concatconcat_0:output:0"#
concat_axisconcat/axis:output:0"
cudnnrnnCudnnRNN:output:0"!

cudnnrnn_0CudnnRNN:output_h:0"!

cudnnrnn_1CudnnRNN:output_c:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"-
cudnnrnn_input_cCudnnRNN/input_c:output:0"!

expanddimsExpandDims:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"
init_hinit_h_0"
reshapeReshape:output:0"
	reshape_1Reshape_1:output:0"!

reshape_10Reshape_10:output:0"!

reshape_11Reshape_11:output:0"!

reshape_12Reshape_12:output:0"
	reshape_2Reshape_2:output:0"
	reshape_3Reshape_3:output:0"
	reshape_4Reshape_4:output:0"
	reshape_5Reshape_5:output:0"
	reshape_6Reshape_6:output:0"
	reshape_7Reshape_7:output:0"
	reshape_8Reshape_8:output:0"
	reshape_9Reshape_9:output:0"
splitsplit_0:output:0"
split_1split_1_0:output:0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0")
strided_slicestrided_slice_0:output:0"
	transposetranspose_0:y:0" 
transpose_1transpose_1_0:y:0"-
transpose_1_permtranspose_1/perm:output:0" 
transpose_2transpose_2_0:y:0"-
transpose_2_permtranspose_2/perm:output:0" 
transpose_3transpose_3_0:y:0"-
transpose_3_permtranspose_3/perm:output:0" 
transpose_4transpose_4_0:y:0"-
transpose_4_permtranspose_4/perm:output:0" 
transpose_5transpose_5_0:y:0"-
transpose_5_permtranspose_5/perm:output:0" 
transpose_6transpose_6_0:y:0"-
transpose_6_permtranspose_6/perm:output:0" 
transpose_7transpose_7_0:y:0"-
transpose_7_permtranspose_7/perm:output:0")
transpose_permtranspose/perm:output:0*\
_input_shapesK
I:??????????:?????????:Z:Z:Z*<
api_implements*(gru_e9d87f5a-6bce-4ea5-b775-fb635bd39d20*
api_preferred_deviceGPU*T
backward_function_name:8__inference___backward_gpu_gru_with_fallback_19488_19706*
go_backwards( *

time_major( :T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinit_h:FB

_output_shapes

:Z
 
_user_specified_namekernel:PL

_output_shapes

:Z
*
_user_specified_namerecurrent_kernel:D@

_output_shapes

:Z

_user_specified_namebias
ڮ
?
8__inference___backward_gpu_gru_with_fallback_15965_16183
placeholder
placeholder_1
placeholder_2
placeholder_3/
+gradients_strided_slice_grad_shape_cudnnrnnB
>gradients_strided_slice_grad_mlcstridedslicegrad_strided_sliceA
=gradients_transpose_7_grad_invertpermutation_transpose_7_perm;
7gradients_transpose_7_grad_mlctransposegrad_transpose_7)
%gradients_squeeze_grad_shape_cudnnrnn!
gradients_zeros_like_cudnnrnn#
gradients_zeros_like_1_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims=
9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c3
/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h)
%gradients_concat_grad_add_concat_axis5
1gradients_reshape_1_grad_mlcreshapegrad_reshape_15
1gradients_reshape_2_grad_mlcreshapegrad_reshape_25
1gradients_reshape_3_grad_mlcreshapegrad_reshape_35
1gradients_reshape_4_grad_mlcreshapegrad_reshape_45
1gradients_reshape_5_grad_mlcreshapegrad_reshape_55
1gradients_reshape_6_grad_mlcreshapegrad_reshape_65
1gradients_reshape_7_grad_mlcreshapegrad_reshape_75
1gradients_reshape_8_grad_mlcreshapegrad_reshape_85
1gradients_reshape_9_grad_mlcreshapegrad_reshape_97
3gradients_reshape_10_grad_mlcreshapegrad_reshape_107
3gradients_reshape_11_grad_mlcreshapegrad_reshape_117
3gradients_reshape_12_grad_mlcreshapegrad_reshape_12A
=gradients_transpose_1_grad_invertpermutation_transpose_1_perm;
7gradients_transpose_1_grad_mlctransposegrad_transpose_1A
=gradients_transpose_2_grad_invertpermutation_transpose_2_perm;
7gradients_transpose_2_grad_mlctransposegrad_transpose_2A
=gradients_transpose_3_grad_invertpermutation_transpose_3_perm;
7gradients_transpose_3_grad_mlctransposegrad_transpose_3A
=gradients_transpose_4_grad_invertpermutation_transpose_4_perm;
7gradients_transpose_4_grad_mlctransposegrad_transpose_4A
=gradients_transpose_5_grad_invertpermutation_transpose_5_perm;
7gradients_transpose_5_grad_mlctransposegrad_transpose_5A
=gradients_transpose_6_grad_invertpermutation_transpose_6_perm;
7gradients_transpose_6_grad_mlctransposegrad_transpose_63
/gradients_split_2_grad_concat_split_2_split_dim5
1gradients_split_grad_mlcsplitgrad_split_split_dim+
'gradients_split_grad_mlcsplitgrad_split9
5gradients_split_1_grad_mlcsplitgrad_split_1_split_dim/
+gradients_split_1_grad_mlcsplitgrad_split_11
-gradients_reshape_grad_mlcreshapegrad_reshape
identity

identity_1

identity_2

identity_3

identity_4?u
gradients/grad_ys_0Identityplaceholder*
T0*'
_output_shapes
:?????????2
gradients/grad_ys_0?
gradients/grad_ys_1Identityplaceholder_1*
T0*4
_output_shapes"
 :??????????????????2
gradients/grad_ys_1w
gradients/grad_ys_2Identityplaceholder_2*
T0*'
_output_shapes
:?????????2
gradients/grad_ys_2f
gradients/grad_ys_3Identityplaceholder_3*
T0*
_output_shapes
: 2
gradients/grad_ys_3?
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:2$
"gradients/strided_slice_grad/Shape?
6gradients/strided_slice_grad/MLCStridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*
valueB:
?????????28
6gradients/strided_slice_grad/MLCStridedSliceGrad/begin?
4gradients/strided_slice_grad/MLCStridedSliceGrad/endConst*
_output_shapes
:*
dtype0*
valueB: 26
4gradients/strided_slice_grad/MLCStridedSliceGrad/end?
8gradients/strided_slice_grad/MLCStridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*
valueB:2:
8gradients/strided_slice_grad/MLCStridedSliceGrad/strides?
0gradients/strided_slice_grad/MLCStridedSliceGradMLCStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0?gradients/strided_slice_grad/MLCStridedSliceGrad/begin:output:0=gradients/strided_slice_grad/MLCStridedSliceGrad/end:output:0Agradients/strided_slice_grad/MLCStridedSliceGrad/strides:output:0gradients/grad_ys_0:output:0>gradients_strided_slice_grad_mlcstridedslicegrad_strided_slice*
Index0*
T0*4
_output_shapes"
 :??????????????????*
shrink_axis_mask22
0gradients/strided_slice_grad/MLCStridedSliceGrad?
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:2.
,gradients/transpose_7_grad/InvertPermutation?
+gradients/transpose_7_grad/MLCTransposeGradMLCTransposeGradgradients/grad_ys_1:output:00gradients/transpose_7_grad/InvertPermutation:y:07gradients_transpose_7_grad_mlctransposegrad_transpose_7*
T0*4
_output_shapes"
 :??????????????????2-
+gradients/transpose_7_grad/MLCTransposeGrad?
gradients/Squeeze_grad/ShapeShape%gradients_squeeze_grad_shape_cudnnrnn*
T0*
_output_shapes
:2
gradients/Squeeze_grad/Shape?
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*+
_output_shapes
:?????????2 
gradients/Squeeze_grad/Reshape?
gradients/AddNAddN9gradients/strided_slice_grad/MLCStridedSliceGrad:output:0/gradients/transpose_7_grad/MLCTransposeGrad:y:0*
N*
T0*C
_class9
75loc:@gradients/strided_slice_grad/MLCStridedSliceGrad*4
_output_shapes"
 :??????????????????2
gradients/AddNy
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
: 2
gradients/zeros_like?
gradients/zeros_like_1	ZerosLikegradients_zeros_like_1_cudnnrnn*
T0*
_output_shapes
:2
gradients/zeros_like_1?
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat+gradients_strided_slice_grad_shape_cudnnrnn%gradients_squeeze_grad_shape_cudnnrnngradients_zeros_like_cudnnrnngradients/AddN:sum:0'gradients/Squeeze_grad/Reshape:output:0gradients/zeros_like:y:0gradients_zeros_like_1_cudnnrnn*
T0*T
_output_shapesB
@:??????????????????:?????????: :?*
rnn_modegru2*
(gradients/CudnnRNN_grad/CudnnRNNBackprop?
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:2,
*gradients/transpose_grad/InvertPermutation?
)gradients/transpose_grad/MLCTransposeGradMLCTransposeGrad9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:02gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose*
T0*4
_output_shapes"
 :??????????????????2+
)gradients/transpose_grad/MLCTransposeGrad?
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
:2!
gradients/ExpandDims_grad/Shape?
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*'
_output_shapes
:?????????2#
!gradients/ExpandDims_grad/Reshape?
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:?2
gradients/concat_grad/Shape|
gradients/concat_grad/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_grad/add/y?
gradients/concat_grad/addAddV2%gradients_concat_grad_add_concat_axis$gradients/concat_grad/add/y:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/add?
)gradients/concat_grad/strided_slice/stackPack%gradients_concat_grad_add_concat_axis*
N*
T0*
_output_shapes
:2+
)gradients/concat_grad/strided_slice/stack?
+gradients/concat_grad/strided_slice/stack_1Packgradients/concat_grad/add:z:0*
N*
T0*
_output_shapes
:2-
+gradients/concat_grad/strided_slice/stack_1?
+gradients/concat_grad/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+gradients/concat_grad/strided_slice/stack_2?
#gradients/concat_grad/strided_sliceStridedSlice$gradients/concat_grad/Shape:output:02gradients/concat_grad/strided_slice/stack:output:04gradients/concat_grad/strided_slice/stack_1:output:04gradients/concat_grad/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#gradients/concat_grad/strided_slice?
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:?2
gradients/concat_grad/Shape_1?
gradients/concat_grad/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_grad/add_1/y?
gradients/concat_grad/add_1AddV2%gradients_concat_grad_add_concat_axis&gradients/concat_grad/add_1/y:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/add_1?
+gradients/concat_grad/strided_slice_1/stackPack%gradients_concat_grad_add_concat_axis*
N*
T0*
_output_shapes
:2-
+gradients/concat_grad/strided_slice_1/stack?
-gradients/concat_grad/strided_slice_1/stack_1Packgradients/concat_grad/add_1:z:0*
N*
T0*
_output_shapes
:2/
-gradients/concat_grad/strided_slice_1/stack_1?
-gradients/concat_grad/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-gradients/concat_grad/strided_slice_1/stack_2?
%gradients/concat_grad/strided_slice_1StridedSlice&gradients/concat_grad/Shape_1:output:04gradients/concat_grad/strided_slice_1/stack:output:06gradients/concat_grad/strided_slice_1/stack_1:output:06gradients/concat_grad/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%gradients/concat_grad/strided_slice_1?
gradients/concat_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:?2
gradients/concat_grad/Shape_2?
gradients/concat_grad/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_grad/add_2/y?
gradients/concat_grad/add_2AddV2%gradients_concat_grad_add_concat_axis&gradients/concat_grad/add_2/y:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/add_2?
+gradients/concat_grad/strided_slice_2/stackPack%gradients_concat_grad_add_concat_axis*
N*
T0*
_output_shapes
:2-
+gradients/concat_grad/strided_slice_2/stack?
-gradients/concat_grad/strided_slice_2/stack_1Packgradients/concat_grad/add_2:z:0*
N*
T0*
_output_shapes
:2/
-gradients/concat_grad/strided_slice_2/stack_1?
-gradients/concat_grad/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-gradients/concat_grad/strided_slice_2/stack_2?
%gradients/concat_grad/strided_slice_2StridedSlice&gradients/concat_grad/Shape_2:output:04gradients/concat_grad/strided_slice_2/stack:output:06gradients/concat_grad/strided_slice_2/stack_1:output:06gradients/concat_grad/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%gradients/concat_grad/strided_slice_2?
gradients/concat_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:?2
gradients/concat_grad/Shape_3?
gradients/concat_grad/add_3/yConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_grad/add_3/y?
gradients/concat_grad/add_3AddV2%gradients_concat_grad_add_concat_axis&gradients/concat_grad/add_3/y:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/add_3?
+gradients/concat_grad/strided_slice_3/stackPack%gradients_concat_grad_add_concat_axis*
N*
T0*
_output_shapes
:2-
+gradients/concat_grad/strided_slice_3/stack?
-gradients/concat_grad/strided_slice_3/stack_1Packgradients/concat_grad/add_3:z:0*
N*
T0*
_output_shapes
:2/
-gradients/concat_grad/strided_slice_3/stack_1?
-gradients/concat_grad/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-gradients/concat_grad/strided_slice_3/stack_2?
%gradients/concat_grad/strided_slice_3StridedSlice&gradients/concat_grad/Shape_3:output:04gradients/concat_grad/strided_slice_3/stack:output:06gradients/concat_grad/strided_slice_3/stack_1:output:06gradients/concat_grad/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%gradients/concat_grad/strided_slice_3?
gradients/concat_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:?2
gradients/concat_grad/Shape_4?
gradients/concat_grad/add_4/yConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_grad/add_4/y?
gradients/concat_grad/add_4AddV2%gradients_concat_grad_add_concat_axis&gradients/concat_grad/add_4/y:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/add_4?
+gradients/concat_grad/strided_slice_4/stackPack%gradients_concat_grad_add_concat_axis*
N*
T0*
_output_shapes
:2-
+gradients/concat_grad/strided_slice_4/stack?
-gradients/concat_grad/strided_slice_4/stack_1Packgradients/concat_grad/add_4:z:0*
N*
T0*
_output_shapes
:2/
-gradients/concat_grad/strided_slice_4/stack_1?
-gradients/concat_grad/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-gradients/concat_grad/strided_slice_4/stack_2?
%gradients/concat_grad/strided_slice_4StridedSlice&gradients/concat_grad/Shape_4:output:04gradients/concat_grad/strided_slice_4/stack:output:06gradients/concat_grad/strided_slice_4/stack_1:output:06gradients/concat_grad/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%gradients/concat_grad/strided_slice_4?
gradients/concat_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:?2
gradients/concat_grad/Shape_5?
gradients/concat_grad/add_5/yConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_grad/add_5/y?
gradients/concat_grad/add_5AddV2%gradients_concat_grad_add_concat_axis&gradients/concat_grad/add_5/y:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/add_5?
+gradients/concat_grad/strided_slice_5/stackPack%gradients_concat_grad_add_concat_axis*
N*
T0*
_output_shapes
:2-
+gradients/concat_grad/strided_slice_5/stack?
-gradients/concat_grad/strided_slice_5/stack_1Packgradients/concat_grad/add_5:z:0*
N*
T0*
_output_shapes
:2/
-gradients/concat_grad/strided_slice_5/stack_1?
-gradients/concat_grad/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-gradients/concat_grad/strided_slice_5/stack_2?
%gradients/concat_grad/strided_slice_5StridedSlice&gradients/concat_grad/Shape_5:output:04gradients/concat_grad/strided_slice_5/stack:output:06gradients/concat_grad/strided_slice_5/stack_1:output:06gradients/concat_grad/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%gradients/concat_grad/strided_slice_5?
gradients/concat_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape_6?
gradients/concat_grad/add_6/yConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_grad/add_6/y?
gradients/concat_grad/add_6AddV2%gradients_concat_grad_add_concat_axis&gradients/concat_grad/add_6/y:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/add_6?
+gradients/concat_grad/strided_slice_6/stackPack%gradients_concat_grad_add_concat_axis*
N*
T0*
_output_shapes
:2-
+gradients/concat_grad/strided_slice_6/stack?
-gradients/concat_grad/strided_slice_6/stack_1Packgradients/concat_grad/add_6:z:0*
N*
T0*
_output_shapes
:2/
-gradients/concat_grad/strided_slice_6/stack_1?
-gradients/concat_grad/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-gradients/concat_grad/strided_slice_6/stack_2?
%gradients/concat_grad/strided_slice_6StridedSlice&gradients/concat_grad/Shape_6:output:04gradients/concat_grad/strided_slice_6/stack:output:06gradients/concat_grad/strided_slice_6/stack_1:output:06gradients/concat_grad/strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%gradients/concat_grad/strided_slice_6?
gradients/concat_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape_7?
gradients/concat_grad/add_7/yConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_grad/add_7/y?
gradients/concat_grad/add_7AddV2%gradients_concat_grad_add_concat_axis&gradients/concat_grad/add_7/y:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/add_7?
+gradients/concat_grad/strided_slice_7/stackPack%gradients_concat_grad_add_concat_axis*
N*
T0*
_output_shapes
:2-
+gradients/concat_grad/strided_slice_7/stack?
-gradients/concat_grad/strided_slice_7/stack_1Packgradients/concat_grad/add_7:z:0*
N*
T0*
_output_shapes
:2/
-gradients/concat_grad/strided_slice_7/stack_1?
-gradients/concat_grad/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-gradients/concat_grad/strided_slice_7/stack_2?
%gradients/concat_grad/strided_slice_7StridedSlice&gradients/concat_grad/Shape_7:output:04gradients/concat_grad/strided_slice_7/stack:output:06gradients/concat_grad/strided_slice_7/stack_1:output:06gradients/concat_grad/strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%gradients/concat_grad/strided_slice_7?
gradients/concat_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape_8?
gradients/concat_grad/add_8/yConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_grad/add_8/y?
gradients/concat_grad/add_8AddV2%gradients_concat_grad_add_concat_axis&gradients/concat_grad/add_8/y:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/add_8?
+gradients/concat_grad/strided_slice_8/stackPack%gradients_concat_grad_add_concat_axis*
N*
T0*
_output_shapes
:2-
+gradients/concat_grad/strided_slice_8/stack?
-gradients/concat_grad/strided_slice_8/stack_1Packgradients/concat_grad/add_8:z:0*
N*
T0*
_output_shapes
:2/
-gradients/concat_grad/strided_slice_8/stack_1?
-gradients/concat_grad/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-gradients/concat_grad/strided_slice_8/stack_2?
%gradients/concat_grad/strided_slice_8StridedSlice&gradients/concat_grad/Shape_8:output:04gradients/concat_grad/strided_slice_8/stack:output:06gradients/concat_grad/strided_slice_8/stack_1:output:06gradients/concat_grad/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%gradients/concat_grad/strided_slice_8?
gradients/concat_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape_9?
gradients/concat_grad/add_9/yConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_grad/add_9/y?
gradients/concat_grad/add_9AddV2%gradients_concat_grad_add_concat_axis&gradients/concat_grad/add_9/y:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/add_9?
+gradients/concat_grad/strided_slice_9/stackPack%gradients_concat_grad_add_concat_axis*
N*
T0*
_output_shapes
:2-
+gradients/concat_grad/strided_slice_9/stack?
-gradients/concat_grad/strided_slice_9/stack_1Packgradients/concat_grad/add_9:z:0*
N*
T0*
_output_shapes
:2/
-gradients/concat_grad/strided_slice_9/stack_1?
-gradients/concat_grad/strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-gradients/concat_grad/strided_slice_9/stack_2?
%gradients/concat_grad/strided_slice_9StridedSlice&gradients/concat_grad/Shape_9:output:04gradients/concat_grad/strided_slice_9/stack:output:06gradients/concat_grad/strided_slice_9/stack_1:output:06gradients/concat_grad/strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%gradients/concat_grad/strided_slice_9?
gradients/concat_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB:2 
gradients/concat_grad/Shape_10?
gradients/concat_grad/add_10/yConst*
_output_shapes
: *
dtype0*
value	B :2 
gradients/concat_grad/add_10/y?
gradients/concat_grad/add_10AddV2%gradients_concat_grad_add_concat_axis'gradients/concat_grad/add_10/y:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/add_10?
,gradients/concat_grad/strided_slice_10/stackPack%gradients_concat_grad_add_concat_axis*
N*
T0*
_output_shapes
:2.
,gradients/concat_grad/strided_slice_10/stack?
.gradients/concat_grad/strided_slice_10/stack_1Pack gradients/concat_grad/add_10:z:0*
N*
T0*
_output_shapes
:20
.gradients/concat_grad/strided_slice_10/stack_1?
.gradients/concat_grad/strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.gradients/concat_grad/strided_slice_10/stack_2?
&gradients/concat_grad/strided_slice_10StridedSlice'gradients/concat_grad/Shape_10:output:05gradients/concat_grad/strided_slice_10/stack:output:07gradients/concat_grad/strided_slice_10/stack_1:output:07gradients/concat_grad/strided_slice_10/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2(
&gradients/concat_grad/strided_slice_10?
gradients/concat_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB:2 
gradients/concat_grad/Shape_11?
gradients/concat_grad/add_11/yConst*
_output_shapes
: *
dtype0*
value	B :2 
gradients/concat_grad/add_11/y?
gradients/concat_grad/add_11AddV2%gradients_concat_grad_add_concat_axis'gradients/concat_grad/add_11/y:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/add_11?
,gradients/concat_grad/strided_slice_11/stackPack%gradients_concat_grad_add_concat_axis*
N*
T0*
_output_shapes
:2.
,gradients/concat_grad/strided_slice_11/stack?
.gradients/concat_grad/strided_slice_11/stack_1Pack gradients/concat_grad/add_11:z:0*
N*
T0*
_output_shapes
:20
.gradients/concat_grad/strided_slice_11/stack_1?
.gradients/concat_grad/strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.gradients/concat_grad/strided_slice_11/stack_2?
&gradients/concat_grad/strided_slice_11StridedSlice'gradients/concat_grad/Shape_11:output:05gradients/concat_grad/strided_slice_11/stack:output:07gradients/concat_grad/strided_slice_11/stack_1:output:07gradients/concat_grad/strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2(
&gradients/concat_grad/strided_slice_11?
gradients/concat_grad/packedPack,gradients/concat_grad/strided_slice:output:0.gradients/concat_grad/strided_slice_1:output:0.gradients/concat_grad/strided_slice_2:output:0.gradients/concat_grad/strided_slice_3:output:0.gradients/concat_grad/strided_slice_4:output:0.gradients/concat_grad/strided_slice_5:output:0.gradients/concat_grad/strided_slice_6:output:0.gradients/concat_grad/strided_slice_7:output:0.gradients/concat_grad/strided_slice_8:output:0.gradients/concat_grad/strided_slice_9:output:0/gradients/concat_grad/strided_slice_10:output:0/gradients/concat_grad/strided_slice_11:output:0*
N*
T0*
_output_shapes
:2
gradients/concat_grad/packed?
%gradients/concat_grad/MLCConcatV2GradMLCConcatV2Grad:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0%gradients/concat_grad/packed:output:0%gradients_concat_grad_add_concat_axis$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0&gradients/concat_grad/Shape_2:output:0&gradients/concat_grad/Shape_3:output:0&gradients/concat_grad/Shape_4:output:0&gradients/concat_grad/Shape_5:output:0&gradients/concat_grad/Shape_6:output:0&gradients/concat_grad/Shape_7:output:0&gradients/concat_grad/Shape_8:output:0&gradients/concat_grad/Shape_9:output:0'gradients/concat_grad/Shape_10:output:0'gradients/concat_grad/Shape_11:output:0/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat*
N*
T0*

Tlen0*b
_output_shapesP
N:?:?:?:?:?:?::::::2'
%gradients/concat_grad/MLCConcatV2Grad?
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
gradients/Reshape_1_grad/Shape?
'gradients/Reshape_1_grad/MLCReshapeGradMLCReshapeGrad.gradients/concat_grad/MLCConcatV2Grad:output:0'gradients/Reshape_1_grad/Shape:output:01gradients_reshape_1_grad_mlcreshapegrad_reshape_1*
T0*
_output_shapes

:2)
'gradients/Reshape_1_grad/MLCReshapeGrad?
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
gradients/Reshape_2_grad/Shape?
'gradients/Reshape_2_grad/MLCReshapeGradMLCReshapeGrad.gradients/concat_grad/MLCConcatV2Grad:output:1'gradients/Reshape_2_grad/Shape:output:01gradients_reshape_2_grad_mlcreshapegrad_reshape_2*
T0*
_output_shapes

:2)
'gradients/Reshape_2_grad/MLCReshapeGrad?
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
gradients/Reshape_3_grad/Shape?
'gradients/Reshape_3_grad/MLCReshapeGradMLCReshapeGrad.gradients/concat_grad/MLCConcatV2Grad:output:2'gradients/Reshape_3_grad/Shape:output:01gradients_reshape_3_grad_mlcreshapegrad_reshape_3*
T0*
_output_shapes

:2)
'gradients/Reshape_3_grad/MLCReshapeGrad?
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
gradients/Reshape_4_grad/Shape?
'gradients/Reshape_4_grad/MLCReshapeGradMLCReshapeGrad.gradients/concat_grad/MLCConcatV2Grad:output:3'gradients/Reshape_4_grad/Shape:output:01gradients_reshape_4_grad_mlcreshapegrad_reshape_4*
T0*
_output_shapes

:2)
'gradients/Reshape_4_grad/MLCReshapeGrad?
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
gradients/Reshape_5_grad/Shape?
'gradients/Reshape_5_grad/MLCReshapeGradMLCReshapeGrad.gradients/concat_grad/MLCConcatV2Grad:output:4'gradients/Reshape_5_grad/Shape:output:01gradients_reshape_5_grad_mlcreshapegrad_reshape_5*
T0*
_output_shapes

:2)
'gradients/Reshape_5_grad/MLCReshapeGrad?
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
gradients/Reshape_6_grad/Shape?
'gradients/Reshape_6_grad/MLCReshapeGradMLCReshapeGrad.gradients/concat_grad/MLCConcatV2Grad:output:5'gradients/Reshape_6_grad/Shape:output:01gradients_reshape_6_grad_mlcreshapegrad_reshape_6*
T0*
_output_shapes

:2)
'gradients/Reshape_6_grad/MLCReshapeGrad?
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2 
gradients/Reshape_7_grad/Shape?
'gradients/Reshape_7_grad/MLCReshapeGradMLCReshapeGrad.gradients/concat_grad/MLCConcatV2Grad:output:6'gradients/Reshape_7_grad/Shape:output:01gradients_reshape_7_grad_mlcreshapegrad_reshape_7*
T0*
_output_shapes
:2)
'gradients/Reshape_7_grad/MLCReshapeGrad?
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2 
gradients/Reshape_8_grad/Shape?
'gradients/Reshape_8_grad/MLCReshapeGradMLCReshapeGrad.gradients/concat_grad/MLCConcatV2Grad:output:7'gradients/Reshape_8_grad/Shape:output:01gradients_reshape_8_grad_mlcreshapegrad_reshape_8*
T0*
_output_shapes
:2)
'gradients/Reshape_8_grad/MLCReshapeGrad?
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2 
gradients/Reshape_9_grad/Shape?
'gradients/Reshape_9_grad/MLCReshapeGradMLCReshapeGrad.gradients/concat_grad/MLCConcatV2Grad:output:8'gradients/Reshape_9_grad/Shape:output:01gradients_reshape_9_grad_mlcreshapegrad_reshape_9*
T0*
_output_shapes
:2)
'gradients/Reshape_9_grad/MLCReshapeGrad?
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2!
gradients/Reshape_10_grad/Shape?
(gradients/Reshape_10_grad/MLCReshapeGradMLCReshapeGrad.gradients/concat_grad/MLCConcatV2Grad:output:9(gradients/Reshape_10_grad/Shape:output:03gradients_reshape_10_grad_mlcreshapegrad_reshape_10*
T0*
_output_shapes
:2*
(gradients/Reshape_10_grad/MLCReshapeGrad?
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2!
gradients/Reshape_11_grad/Shape?
(gradients/Reshape_11_grad/MLCReshapeGradMLCReshapeGrad/gradients/concat_grad/MLCConcatV2Grad:output:10(gradients/Reshape_11_grad/Shape:output:03gradients_reshape_11_grad_mlcreshapegrad_reshape_11*
T0*
_output_shapes
:2*
(gradients/Reshape_11_grad/MLCReshapeGrad?
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2!
gradients/Reshape_12_grad/Shape?
(gradients/Reshape_12_grad/MLCReshapeGradMLCReshapeGrad/gradients/concat_grad/MLCConcatV2Grad:output:11(gradients/Reshape_12_grad/Shape:output:03gradients_reshape_12_grad_mlcreshapegrad_reshape_12*
T0*
_output_shapes
:2*
(gradients/Reshape_12_grad/MLCReshapeGrad?
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:2.
,gradients/transpose_1_grad/InvertPermutation?
+gradients/transpose_1_grad/MLCTransposeGradMLCTransposeGrad0gradients/Reshape_1_grad/MLCReshapeGrad:output:00gradients/transpose_1_grad/InvertPermutation:y:07gradients_transpose_1_grad_mlctransposegrad_transpose_1*
T0*
_output_shapes

:2-
+gradients/transpose_1_grad/MLCTransposeGrad?
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:2.
,gradients/transpose_2_grad/InvertPermutation?
+gradients/transpose_2_grad/MLCTransposeGradMLCTransposeGrad0gradients/Reshape_2_grad/MLCReshapeGrad:output:00gradients/transpose_2_grad/InvertPermutation:y:07gradients_transpose_2_grad_mlctransposegrad_transpose_2*
T0*
_output_shapes

:2-
+gradients/transpose_2_grad/MLCTransposeGrad?
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:2.
,gradients/transpose_3_grad/InvertPermutation?
+gradients/transpose_3_grad/MLCTransposeGradMLCTransposeGrad0gradients/Reshape_3_grad/MLCReshapeGrad:output:00gradients/transpose_3_grad/InvertPermutation:y:07gradients_transpose_3_grad_mlctransposegrad_transpose_3*
T0*
_output_shapes

:2-
+gradients/transpose_3_grad/MLCTransposeGrad?
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:2.
,gradients/transpose_4_grad/InvertPermutation?
+gradients/transpose_4_grad/MLCTransposeGradMLCTransposeGrad0gradients/Reshape_4_grad/MLCReshapeGrad:output:00gradients/transpose_4_grad/InvertPermutation:y:07gradients_transpose_4_grad_mlctransposegrad_transpose_4*
T0*
_output_shapes

:2-
+gradients/transpose_4_grad/MLCTransposeGrad?
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:2.
,gradients/transpose_5_grad/InvertPermutation?
+gradients/transpose_5_grad/MLCTransposeGradMLCTransposeGrad0gradients/Reshape_5_grad/MLCReshapeGrad:output:00gradients/transpose_5_grad/InvertPermutation:y:07gradients_transpose_5_grad_mlctransposegrad_transpose_5*
T0*
_output_shapes

:2-
+gradients/transpose_5_grad/MLCTransposeGrad?
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:2.
,gradients/transpose_6_grad/InvertPermutation?
+gradients/transpose_6_grad/MLCTransposeGradMLCTransposeGrad0gradients/Reshape_6_grad/MLCReshapeGrad:output:00gradients/transpose_6_grad/InvertPermutation:y:07gradients_transpose_6_grad_mlctransposegrad_transpose_6*
T0*
_output_shapes

:2-
+gradients/transpose_6_grad/MLCTransposeGrad?
gradients/split_2_grad/concatConcatV20gradients/Reshape_7_grad/MLCReshapeGrad:output:00gradients/Reshape_8_grad/MLCReshapeGrad:output:00gradients/Reshape_9_grad/MLCReshapeGrad:output:01gradients/Reshape_10_grad/MLCReshapeGrad:output:01gradients/Reshape_11_grad/MLCReshapeGrad:output:01gradients/Reshape_12_grad/MLCReshapeGrad:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes	
:?2
gradients/split_2_grad/concat?
!gradients/split_grad/MLCSplitGradMLCSplitGrad/gradients/transpose_1_grad/MLCTransposeGrad:y:0/gradients/transpose_2_grad/MLCTransposeGrad:y:0/gradients/transpose_3_grad/MLCTransposeGrad:y:01gradients_split_grad_mlcsplitgrad_split_split_dim'gradients_split_grad_mlcsplitgrad_split*
N*
T0*
_output_shapes

:Z2#
!gradients/split_grad/MLCSplitGrad?
#gradients/split_1_grad/MLCSplitGradMLCSplitGrad/gradients/transpose_4_grad/MLCTransposeGrad:y:0/gradients/transpose_5_grad/MLCTransposeGrad:y:0/gradients/transpose_6_grad/MLCTransposeGrad:y:05gradients_split_1_grad_mlcsplitgrad_split_1_split_dim+gradients_split_1_grad_mlcsplitgrad_split_1*
N*
T0*
_output_shapes

:Z2%
#gradients/split_1_grad/MLCSplitGrad?
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"   Z   2
gradients/Reshape_grad/Shape?
%gradients/Reshape_grad/MLCReshapeGradMLCReshapeGrad&gradients/split_2_grad/concat:output:0%gradients/Reshape_grad/Shape:output:0-gradients_reshape_grad_mlcreshapegrad_reshape*
T0*
_output_shapes

:Z2'
%gradients/Reshape_grad/MLCReshapeGrad?
IdentityIdentity-gradients/transpose_grad/MLCTransposeGrad:y:0*
T0*4
_output_shapes"
 :??????????????????2

Identity?

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0*
T0*'
_output_shapes
:?????????2

Identity_1y

Identity_2Identity*gradients/split_grad/MLCSplitGrad:output:0*
T0*
_output_shapes

:Z2

Identity_2{

Identity_3Identity,gradients/split_1_grad/MLCSplitGrad:output:0*
T0*
_output_shapes

:Z2

Identity_3}

Identity_4Identity.gradients/Reshape_grad/MLCReshapeGrad:output:0*
T0*
_output_shapes

:Z2

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*?
_input_shapes?
?:?????????:??????????????????:?????????: :??????????????????:?????????::??????????????????:?????????: ::??????????????????:?????????: :?::?????????: :?:?:?:?:?:?::::::::::::::::::: : :: ::?*<
api_implements*(gru_d320c0db-d27b-490e-8348-8d4a20a9c88e*
api_preferred_deviceGPU*@
forward_function_name'%__forward_gpu_gru_with_fallback_16182*
go_backwards( *

time_major( :- )
'
_output_shapes
:?????????::6
4
_output_shapes"
 :??????????????????:-)
'
_output_shapes
:?????????:

_output_shapes
: ::6
4
_output_shapes"
 :??????????????????:-)
'
_output_shapes
:?????????: 

_output_shapes
:::6
4
_output_shapes"
 :??????????????????:1-
+
_output_shapes
:?????????:	

_output_shapes
: :


_output_shapes
:::6
4
_output_shapes"
 :??????????????????:1-
+
_output_shapes
:?????????:

_output_shapes
: :!

_output_shapes	
:?: 

_output_shapes
::-)
'
_output_shapes
:?????????:

_output_shapes
: :!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::$ 

_output_shapes

::  

_output_shapes
::$! 

_output_shapes

:: "

_output_shapes
::$# 

_output_shapes

:: $

_output_shapes
::$% 

_output_shapes

:: &

_output_shapes
::$' 

_output_shapes

:: (

_output_shapes
::$) 

_output_shapes

::*

_output_shapes
: :+

_output_shapes
: :$, 

_output_shapes

::-

_output_shapes
: :$. 

_output_shapes

::!/

_output_shapes	
:?
?M
?
%__forward_gpu_gru_with_fallback_18239

inputs
init_h_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3
cudnnrnn
strided_slice
transpose_7_perm
transpose_7

cudnnrnn_0

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
cudnnrnn_input_c

concat
transpose_perm

init_h
concat_axis
	reshape_1
	reshape_2
	reshape_3
	reshape_4
	reshape_5
	reshape_6
	reshape_7
	reshape_8
	reshape_9

reshape_10

reshape_11

reshape_12
transpose_1_perm
transpose_1
transpose_2_perm
transpose_2
transpose_3_perm
transpose_3
transpose_4_perm
transpose_4
transpose_5_perm
transpose_5
transpose_6_perm
transpose_6
split_2_split_dim
split_split_dim	
split
split_1_split_dim
split_1
reshape?u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permO
transpose_0	Transposeinputstranspose/perm:output:0*
T02
	transposeb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims/dim

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2

ExpandDimsP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dimU
split_0Splitsplit/split_dim:output:0kernel*
T0*
	num_split2
splitT
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dime
	split_1_0Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*
	num_split2	
split_1q
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
Reshape/shapea
ReshapeReshapebiasReshape/shape:output:0*
T0*
_output_shapes	
:?2	
ReshapeT
Const_2Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_2h
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_2/split_dim?
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*8
_output_shapes&
$::::::*
	num_split2	
split_2e
Const_3Const*
_output_shapes
:*
dtype0*
valueB:
?????????2	
Const_3u
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_1/perm_
transpose_1_0	Transposesplit_0:output:0transpose_1/perm:output:0*
T02
transpose_1l
	Reshape_1Reshapetranspose_1_0:y:0Const_3:output:0*
T0*
_output_shapes	
:?2
	Reshape_1u
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_2/perm_
transpose_2_0	Transposesplit_0:output:1transpose_2/perm:output:0*
T02
transpose_2l
	Reshape_2Reshapetranspose_2_0:y:0Const_3:output:0*
T0*
_output_shapes	
:?2
	Reshape_2u
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_3/perm_
transpose_3_0	Transposesplit_0:output:2transpose_3/perm:output:0*
T02
transpose_3l
	Reshape_3Reshapetranspose_3_0:y:0Const_3:output:0*
T0*
_output_shapes	
:?2
	Reshape_3u
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_4/perma
transpose_4_0	Transposesplit_1_0:output:0transpose_4/perm:output:0*
T02
transpose_4l
	Reshape_4Reshapetranspose_4_0:y:0Const_3:output:0*
T0*
_output_shapes	
:?2
	Reshape_4u
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_5/perma
transpose_5_0	Transposesplit_1_0:output:1transpose_5/perm:output:0*
T02
transpose_5l
	Reshape_5Reshapetranspose_5_0:y:0Const_3:output:0*
T0*
_output_shapes	
:?2
	Reshape_5u
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_6/perma
transpose_6_0	Transposesplit_1_0:output:2transpose_6/perm:output:0*
T02
transpose_6l
	Reshape_6Reshapetranspose_6_0:y:0Const_3:output:0*
T0*
_output_shapes	
:?2
	Reshape_6j
	Reshape_7Reshapesplit_2:output:0Const_3:output:0*
T0*
_output_shapes
:2
	Reshape_7j
	Reshape_8Reshapesplit_2:output:1Const_3:output:0*
T0*
_output_shapes
:2
	Reshape_8j
	Reshape_9Reshapesplit_2:output:2Const_3:output:0*
T0*
_output_shapes
:2
	Reshape_9l

Reshape_10Reshapesplit_2:output:3Const_3:output:0*
T0*
_output_shapes
:2

Reshape_10l

Reshape_11Reshapesplit_2:output:4Const_3:output:0*
T0*
_output_shapes
:2

Reshape_11l

Reshape_12Reshapesplit_2:output:5Const_3:output:0*
T0*
_output_shapes
:2

Reshape_12\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat/axis?
concat_0ConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
N*
T02
concati
CudnnRNN/input_cConst*
_output_shapes
: *
dtype0*
valueB
 *    2
CudnnRNN/input_c?
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat_0:output:0*
T0*I
_output_shapes7
5:??????????:?????????: :*
rnn_modegru2

CudnnRNN}
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_slice_0StridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
shrink_axis_mask2
strided_slicey
transpose_7/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_7/perm`
transpose_7_0	TransposeCudnnRNN:output:0transpose_7/perm:output:0*
T02
transpose_7{
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:?????????*
squeeze_dims
 2	
Squeezef
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @2	
runtimel
IdentityIdentitystrided_slice_0:output:0*
T0*'
_output_shapes
:?????????2

Identityn

Identity_1Identitytranspose_7_0:y:0*
T0*,
_output_shapes
:??????????2

Identity_1h

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:?????????2

Identity_2W

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: 2

Identity_3"
concatconcat_0:output:0"#
concat_axisconcat/axis:output:0"
cudnnrnnCudnnRNN:output:0"!

cudnnrnn_0CudnnRNN:output_h:0"!

cudnnrnn_1CudnnRNN:output_c:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"-
cudnnrnn_input_cCudnnRNN/input_c:output:0"!

expanddimsExpandDims:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"
init_hinit_h_0"
reshapeReshape:output:0"
	reshape_1Reshape_1:output:0"!

reshape_10Reshape_10:output:0"!

reshape_11Reshape_11:output:0"!

reshape_12Reshape_12:output:0"
	reshape_2Reshape_2:output:0"
	reshape_3Reshape_3:output:0"
	reshape_4Reshape_4:output:0"
	reshape_5Reshape_5:output:0"
	reshape_6Reshape_6:output:0"
	reshape_7Reshape_7:output:0"
	reshape_8Reshape_8:output:0"
	reshape_9Reshape_9:output:0"
splitsplit_0:output:0"
split_1split_1_0:output:0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0")
strided_slicestrided_slice_0:output:0"
	transposetranspose_0:y:0" 
transpose_1transpose_1_0:y:0"-
transpose_1_permtranspose_1/perm:output:0" 
transpose_2transpose_2_0:y:0"-
transpose_2_permtranspose_2/perm:output:0" 
transpose_3transpose_3_0:y:0"-
transpose_3_permtranspose_3/perm:output:0" 
transpose_4transpose_4_0:y:0"-
transpose_4_permtranspose_4/perm:output:0" 
transpose_5transpose_5_0:y:0"-
transpose_5_permtranspose_5/perm:output:0" 
transpose_6transpose_6_0:y:0"-
transpose_6_permtranspose_6/perm:output:0" 
transpose_7transpose_7_0:y:0"-
transpose_7_permtranspose_7/perm:output:0")
transpose_permtranspose/perm:output:0*\
_input_shapesK
I:??????????:?????????:Z:Z:Z*<
api_implements*(gru_2db1bf7b-7db5-4f71-98ab-d0201f7417d2*
api_preferred_deviceGPU*T
backward_function_name:8__inference___backward_gpu_gru_with_fallback_18022_18240*
go_backwards( *

time_major( :T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinit_h:FB

_output_shapes

:Z
 
_user_specified_namekernel:PL

_output_shapes

:Z
*
_user_specified_namerecurrent_kernel:D@

_output_shapes

:Z

_user_specified_namebias
?
?
>__inference_gru_layer_call_and_return_conditional_losses_18748
inputs_0 
read_readvariableop_resource"
read_1_readvariableop_resource"
read_2_readvariableop_resource

identity_3??Read/ReadVariableOp?Read_1/ReadVariableOp?Read_2/ReadVariableOpF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????2
zeros?
Read/ReadVariableOpReadVariableOpread_readvariableop_resource*
_output_shapes

:Z*
dtype02
Read/ReadVariableOpf
IdentityIdentityRead/ReadVariableOp:value:0*
T0*
_output_shapes

:Z2

Identity?
Read_1/ReadVariableOpReadVariableOpread_1_readvariableop_resource*
_output_shapes

:Z*
dtype02
Read_1/ReadVariableOpl

Identity_1IdentityRead_1/ReadVariableOp:value:0*
T0*
_output_shapes

:Z2

Identity_1?
Read_2/ReadVariableOpReadVariableOpread_2_readvariableop_resource*
_output_shapes

:Z*
dtype02
Read_2/ReadVariableOpl

Identity_2IdentityRead_2/ReadVariableOp:value:0*
T0*
_output_shapes

:Z2

Identity_2?
PartitionedCallPartitionedCallinputs_0zeros:output:0Identity:output:0Identity_1:output:0Identity_2:output:0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *\
_output_shapesJ
H:?????????:??????????????????:?????????: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *'
f"R 
__inference_standard_gru_184482
PartitionedCall?

Identity_3IdentityPartitionedCall:output:0^Read/ReadVariableOp^Read_1/ReadVariableOp^Read_2/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_3"!

identity_3Identity_3:output:0*?
_input_shapes.
,:??????????????????:::2*
Read/ReadVariableOpRead/ReadVariableOp2.
Read_1/ReadVariableOpRead_1/ReadVariableOp2.
Read_2/ReadVariableOpRead_2/ReadVariableOp:^ Z
4
_output_shapes"
 :??????????????????
"
_user_specified_name
inputs/0
?<
?
'__inference_gpu_gru_with_fallback_18021

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3?u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm{
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:??????????2
	transposeb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims/dim}

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2

ExpandDimsP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
splitSplitsplit/split_dim:output:0kernel*
T0*2
_output_shapes 
:::*
	num_split2
splitT
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dim?
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*2
_output_shapes 
:::*
	num_split2	
split_1q
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
Reshape/shapea
ReshapeReshapebiasReshape/shape:output:0*
T0*
_output_shapes	
:?2	
ReshapeT
Const_2Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_2h
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_2/split_dim?
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*8
_output_shapes&
$::::::*
	num_split2	
split_2e
Const_3Const*
_output_shapes
:*
dtype0*
valueB:
?????????2	
Const_3u
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_1/perm{
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes

:2
transpose_1j
	Reshape_1Reshapetranspose_1:y:0Const_3:output:0*
T0*
_output_shapes	
:?2
	Reshape_1u
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_2/perm{
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes

:2
transpose_2j
	Reshape_2Reshapetranspose_2:y:0Const_3:output:0*
T0*
_output_shapes	
:?2
	Reshape_2u
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_3/perm{
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

:2
transpose_3j
	Reshape_3Reshapetranspose_3:y:0Const_3:output:0*
T0*
_output_shapes	
:?2
	Reshape_3u
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_4/perm}
transpose_4	Transposesplit_1:output:0transpose_4/perm:output:0*
T0*
_output_shapes

:2
transpose_4j
	Reshape_4Reshapetranspose_4:y:0Const_3:output:0*
T0*
_output_shapes	
:?2
	Reshape_4u
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_5/perm}
transpose_5	Transposesplit_1:output:1transpose_5/perm:output:0*
T0*
_output_shapes

:2
transpose_5j
	Reshape_5Reshapetranspose_5:y:0Const_3:output:0*
T0*
_output_shapes	
:?2
	Reshape_5u
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_6/perm}
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T0*
_output_shapes

:2
transpose_6j
	Reshape_6Reshapetranspose_6:y:0Const_3:output:0*
T0*
_output_shapes	
:?2
	Reshape_6j
	Reshape_7Reshapesplit_2:output:0Const_3:output:0*
T0*
_output_shapes
:2
	Reshape_7j
	Reshape_8Reshapesplit_2:output:1Const_3:output:0*
T0*
_output_shapes
:2
	Reshape_8j
	Reshape_9Reshapesplit_2:output:2Const_3:output:0*
T0*
_output_shapes
:2
	Reshape_9l

Reshape_10Reshapesplit_2:output:3Const_3:output:0*
T0*
_output_shapes
:2

Reshape_10l

Reshape_11Reshapesplit_2:output:4Const_3:output:0*
T0*
_output_shapes
:2

Reshape_11l

Reshape_12Reshapesplit_2:output:5Const_3:output:0*
T0*
_output_shapes
:2

Reshape_12\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat/axis?
concatConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
N*
T0*
_output_shapes	
:?2
concati
CudnnRNN/input_cConst*
_output_shapes
: *
dtype0*
valueB
 *    2
CudnnRNN/input_c?
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat:output:0*
T0*I
_output_shapes7
5:??????????:?????????: :*
rnn_modegru2

CudnnRNN}
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slicey
transpose_7/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_7/perm?
transpose_7	TransposeCudnnRNN:output:0transpose_7/perm:output:0*
T0*,
_output_shapes
:??????????2
transpose_7{
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:?????????*
squeeze_dims
 2	
Squeezef
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @2	
runtimej
IdentityIdentitystrided_slice:output:0*
T0*'
_output_shapes
:?????????2

Identityl

Identity_1Identitytranspose_7:y:0*
T0*,
_output_shapes
:??????????2

Identity_1h

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:?????????2

Identity_2W

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: 2

Identity_3"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*\
_input_shapesK
I:??????????:?????????:Z:Z:Z*<
api_implements*(gru_2db1bf7b-7db5-4f71-98ab-d0201f7417d2*
api_preferred_deviceGPU*
go_backwards( *

time_major( :T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinit_h:FB

_output_shapes

:Z
 
_user_specified_namekernel:PL

_output_shapes

:Z
*
_user_specified_namerecurrent_kernel:D@

_output_shapes

:Z

_user_specified_namebias
?
?
G__inference_sequential_1_layer_call_and_return_conditional_losses_17209
	gru_input
	gru_17196
	gru_17198
	gru_17200
dense_2_17203
dense_2_17205
identity??dense_2/StatefulPartitionedCall?gru/StatefulPartitionedCall?
gru/StatefulPartitionedCallStatefulPartitionedCall	gru_input	gru_17196	gru_17198	gru_17200*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *G
fBR@
>__inference_gru_layer_call_and_return_conditional_losses_171352
gru/StatefulPartitionedCall?
dense_2/StatefulPartitionedCallStatefulPartitionedCall$gru/StatefulPartitionedCall:output:0dense_2_17203dense_2_17205*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_171762!
dense_2/StatefulPartitionedCall?
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0 ^dense_2/StatefulPartitionedCall^gru/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????:::::2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2:
gru/StatefulPartitionedCallgru/StatefulPartitionedCall:W S
,
_output_shapes
:??????????
#
_user_specified_name	gru_input
?2
?
while_body_19786
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_matmul_kernel_0
while_biasadd_unstack_0%
!while_matmul_1_recurrent_kernel_0
while_biasadd_1_unstack_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_matmul_kernel
while_biasadd_unstack#
while_matmul_1_recurrent_kernel
while_biasadd_1_unstack?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
while/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while_matmul_kernel_0*
T0*'
_output_shapes
:?????????Z2
while/MatMul?
while/BiasAddBiasAddwhile/MatMul:product:0while_biasadd_unstack_0*
T0*'
_output_shapes
:?????????Z2
while/BiasAdd\
while/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/Constp
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
while/split/split_dim?
while/splitSplitwhile/split/split_dim:output:0while/BiasAdd:output:0*
T0*M
_output_shapes;
9:?????????:?????????:?????????*
	num_split2
while/split?
while/MatMul_1MatMulwhile_placeholder_2!while_matmul_1_recurrent_kernel_0*
T0*'
_output_shapes
:?????????Z2
while/MatMul_1?
while/BiasAdd_1BiasAddwhile/MatMul_1:product:0while_biasadd_1_unstack_0*
T0*'
_output_shapes
:?????????Z2
while/BiasAdd_1`
while/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
while/Const_1t
while/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
while/split_1/split_dim?
while/split_1Split while/split_1/split_dim:output:0while/BiasAdd_1:output:0*
T0*M
_output_shapes;
9:?????????:?????????:?????????*
	num_split2
while/split_1
	while/addAddV2while/split:output:0while/split_1:output:0*
T0*'
_output_shapes
:?????????2
	while/addj
while/SigmoidSigmoidwhile/add:z:0*
T0*'
_output_shapes
:?????????2
while/Sigmoid?
while/add_1AddV2while/split:output:1while/split_1:output:1*
T0*'
_output_shapes
:?????????2
while/add_1p
while/Sigmoid_1Sigmoidwhile/add_1:z:0*
T0*'
_output_shapes
:?????????2
while/Sigmoid_1|
	while/mulMulwhile/Sigmoid_1:y:0while/split_1:output:2*
T0*'
_output_shapes
:?????????2
	while/mulz
while/add_2AddV2while/split:output:2while/mul:z:0*
T0*'
_output_shapes
:?????????2
while/add_2c

while/TanhTanhwhile/add_2:z:0*
T0*'
_output_shapes
:?????????2

while/Tanh{
while/mul_1Mulwhile/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:?????????2
while/mul_1_
while/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
while/sub/xx
	while/subSubwhile/sub/x:output:0while/Sigmoid:y:0*
T0*'
_output_shapes
:?????????2
	while/subr
while/mul_2Mulwhile/sub:z:0while/Tanh:y:0*
T0*'
_output_shapes
:?????????2
while/mul_2w
while/add_3AddV2while/mul_1:z:0while/mul_2:z:0*
T0*'
_output_shapes
:?????????2
while/add_3?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/add_3:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem`
while/add_4/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_4/yo
while/add_4AddV2while_placeholderwhile/add_4/y:output:0*
T0*
_output_shapes
: 2
while/add_4`
while/add_5/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_5/yv
while/add_5AddV2while_while_loop_counterwhile/add_5/y:output:0*
T0*
_output_shapes
: 2
while/add_5^
while/IdentityIdentitywhile/add_5:z:0*
T0*
_output_shapes
: 2
while/Identityq
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: 2
while/Identity_1b
while/Identity_2Identitywhile/add_4:z:0*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3s
while/Identity_4Identitywhile/add_3:z:0*
T0*'
_output_shapes
:?????????2
while/Identity_4"4
while_biasadd_1_unstackwhile_biasadd_1_unstack_0"0
while_biasadd_unstackwhile_biasadd_unstack_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"D
while_matmul_1_recurrent_kernel!while_matmul_1_recurrent_kernel_0",
while_matmul_kernelwhile_matmul_kernel_0",
while_strided_slicewhile_strided_slice_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*R
_input_shapesA
?: : : : :?????????: : :Z:Z:Z:Z: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:Z: 

_output_shapes
:Z:$	 

_output_shapes

:Z: 


_output_shapes
:Z
??
?
8__inference___backward_gpu_gru_with_fallback_17546_17764
placeholder
placeholder_1
placeholder_2
placeholder_3/
+gradients_strided_slice_grad_shape_cudnnrnnB
>gradients_strided_slice_grad_mlcstridedslicegrad_strided_sliceA
=gradients_transpose_7_grad_invertpermutation_transpose_7_perm;
7gradients_transpose_7_grad_mlctransposegrad_transpose_7)
%gradients_squeeze_grad_shape_cudnnrnn!
gradients_zeros_like_cudnnrnn#
gradients_zeros_like_1_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims=
9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c3
/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h)
%gradients_concat_grad_add_concat_axis5
1gradients_reshape_1_grad_mlcreshapegrad_reshape_15
1gradients_reshape_2_grad_mlcreshapegrad_reshape_25
1gradients_reshape_3_grad_mlcreshapegrad_reshape_35
1gradients_reshape_4_grad_mlcreshapegrad_reshape_45
1gradients_reshape_5_grad_mlcreshapegrad_reshape_55
1gradients_reshape_6_grad_mlcreshapegrad_reshape_65
1gradients_reshape_7_grad_mlcreshapegrad_reshape_75
1gradients_reshape_8_grad_mlcreshapegrad_reshape_85
1gradients_reshape_9_grad_mlcreshapegrad_reshape_97
3gradients_reshape_10_grad_mlcreshapegrad_reshape_107
3gradients_reshape_11_grad_mlcreshapegrad_reshape_117
3gradients_reshape_12_grad_mlcreshapegrad_reshape_12A
=gradients_transpose_1_grad_invertpermutation_transpose_1_perm;
7gradients_transpose_1_grad_mlctransposegrad_transpose_1A
=gradients_transpose_2_grad_invertpermutation_transpose_2_perm;
7gradients_transpose_2_grad_mlctransposegrad_transpose_2A
=gradients_transpose_3_grad_invertpermutation_transpose_3_perm;
7gradients_transpose_3_grad_mlctransposegrad_transpose_3A
=gradients_transpose_4_grad_invertpermutation_transpose_4_perm;
7gradients_transpose_4_grad_mlctransposegrad_transpose_4A
=gradients_transpose_5_grad_invertpermutation_transpose_5_perm;
7gradients_transpose_5_grad_mlctransposegrad_transpose_5A
=gradients_transpose_6_grad_invertpermutation_transpose_6_perm;
7gradients_transpose_6_grad_mlctransposegrad_transpose_63
/gradients_split_2_grad_concat_split_2_split_dim5
1gradients_split_grad_mlcsplitgrad_split_split_dim+
'gradients_split_grad_mlcsplitgrad_split9
5gradients_split_1_grad_mlcsplitgrad_split_1_split_dim/
+gradients_split_1_grad_mlcsplitgrad_split_11
-gradients_reshape_grad_mlcreshapegrad_reshape
identity

identity_1

identity_2

identity_3

identity_4?u
gradients/grad_ys_0Identityplaceholder*
T0*'
_output_shapes
:?????????2
gradients/grad_ys_0|
gradients/grad_ys_1Identityplaceholder_1*
T0*,
_output_shapes
:??????????2
gradients/grad_ys_1w
gradients/grad_ys_2Identityplaceholder_2*
T0*'
_output_shapes
:?????????2
gradients/grad_ys_2f
gradients/grad_ys_3Identityplaceholder_3*
T0*
_output_shapes
: 2
gradients/grad_ys_3?
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:2$
"gradients/strided_slice_grad/Shape?
6gradients/strided_slice_grad/MLCStridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*
valueB:
?????????28
6gradients/strided_slice_grad/MLCStridedSliceGrad/begin?
4gradients/strided_slice_grad/MLCStridedSliceGrad/endConst*
_output_shapes
:*
dtype0*
valueB: 26
4gradients/strided_slice_grad/MLCStridedSliceGrad/end?
8gradients/strided_slice_grad/MLCStridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*
valueB:2:
8gradients/strided_slice_grad/MLCStridedSliceGrad/strides?
0gradients/strided_slice_grad/MLCStridedSliceGradMLCStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0?gradients/strided_slice_grad/MLCStridedSliceGrad/begin:output:0=gradients/strided_slice_grad/MLCStridedSliceGrad/end:output:0Agradients/strided_slice_grad/MLCStridedSliceGrad/strides:output:0gradients/grad_ys_0:output:0>gradients_strided_slice_grad_mlcstridedslicegrad_strided_slice*
Index0*
T0*,
_output_shapes
:??????????*
shrink_axis_mask22
0gradients/strided_slice_grad/MLCStridedSliceGrad?
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:2.
,gradients/transpose_7_grad/InvertPermutation?
+gradients/transpose_7_grad/MLCTransposeGradMLCTransposeGradgradients/grad_ys_1:output:00gradients/transpose_7_grad/InvertPermutation:y:07gradients_transpose_7_grad_mlctransposegrad_transpose_7*
T0*,
_output_shapes
:??????????2-
+gradients/transpose_7_grad/MLCTransposeGrad?
gradients/Squeeze_grad/ShapeShape%gradients_squeeze_grad_shape_cudnnrnn*
T0*
_output_shapes
:2
gradients/Squeeze_grad/Shape?
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*+
_output_shapes
:?????????2 
gradients/Squeeze_grad/Reshape?
gradients/AddNAddN9gradients/strided_slice_grad/MLCStridedSliceGrad:output:0/gradients/transpose_7_grad/MLCTransposeGrad:y:0*
N*
T0*C
_class9
75loc:@gradients/strided_slice_grad/MLCStridedSliceGrad*,
_output_shapes
:??????????2
gradients/AddNy
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
: 2
gradients/zeros_like?
gradients/zeros_like_1	ZerosLikegradients_zeros_like_1_cudnnrnn*
T0*
_output_shapes
:2
gradients/zeros_like_1?
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat+gradients_strided_slice_grad_shape_cudnnrnn%gradients_squeeze_grad_shape_cudnnrnngradients_zeros_like_cudnnrnngradients/AddN:sum:0'gradients/Squeeze_grad/Reshape:output:0gradients/zeros_like:y:0gradients_zeros_like_1_cudnnrnn*
T0*L
_output_shapes:
8:??????????:?????????: :?*
rnn_modegru2*
(gradients/CudnnRNN_grad/CudnnRNNBackprop?
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:2,
*gradients/transpose_grad/InvertPermutation?
)gradients/transpose_grad/MLCTransposeGradMLCTransposeGrad9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:02gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose*
T0*,
_output_shapes
:??????????2+
)gradients/transpose_grad/MLCTransposeGrad?
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
:2!
gradients/ExpandDims_grad/Shape?
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*'
_output_shapes
:?????????2#
!gradients/ExpandDims_grad/Reshape?
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:?2
gradients/concat_grad/Shape|
gradients/concat_grad/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_grad/add/y?
gradients/concat_grad/addAddV2%gradients_concat_grad_add_concat_axis$gradients/concat_grad/add/y:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/add?
)gradients/concat_grad/strided_slice/stackPack%gradients_concat_grad_add_concat_axis*
N*
T0*
_output_shapes
:2+
)gradients/concat_grad/strided_slice/stack?
+gradients/concat_grad/strided_slice/stack_1Packgradients/concat_grad/add:z:0*
N*
T0*
_output_shapes
:2-
+gradients/concat_grad/strided_slice/stack_1?
+gradients/concat_grad/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+gradients/concat_grad/strided_slice/stack_2?
#gradients/concat_grad/strided_sliceStridedSlice$gradients/concat_grad/Shape:output:02gradients/concat_grad/strided_slice/stack:output:04gradients/concat_grad/strided_slice/stack_1:output:04gradients/concat_grad/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#gradients/concat_grad/strided_slice?
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:?2
gradients/concat_grad/Shape_1?
gradients/concat_grad/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_grad/add_1/y?
gradients/concat_grad/add_1AddV2%gradients_concat_grad_add_concat_axis&gradients/concat_grad/add_1/y:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/add_1?
+gradients/concat_grad/strided_slice_1/stackPack%gradients_concat_grad_add_concat_axis*
N*
T0*
_output_shapes
:2-
+gradients/concat_grad/strided_slice_1/stack?
-gradients/concat_grad/strided_slice_1/stack_1Packgradients/concat_grad/add_1:z:0*
N*
T0*
_output_shapes
:2/
-gradients/concat_grad/strided_slice_1/stack_1?
-gradients/concat_grad/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-gradients/concat_grad/strided_slice_1/stack_2?
%gradients/concat_grad/strided_slice_1StridedSlice&gradients/concat_grad/Shape_1:output:04gradients/concat_grad/strided_slice_1/stack:output:06gradients/concat_grad/strided_slice_1/stack_1:output:06gradients/concat_grad/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%gradients/concat_grad/strided_slice_1?
gradients/concat_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:?2
gradients/concat_grad/Shape_2?
gradients/concat_grad/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_grad/add_2/y?
gradients/concat_grad/add_2AddV2%gradients_concat_grad_add_concat_axis&gradients/concat_grad/add_2/y:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/add_2?
+gradients/concat_grad/strided_slice_2/stackPack%gradients_concat_grad_add_concat_axis*
N*
T0*
_output_shapes
:2-
+gradients/concat_grad/strided_slice_2/stack?
-gradients/concat_grad/strided_slice_2/stack_1Packgradients/concat_grad/add_2:z:0*
N*
T0*
_output_shapes
:2/
-gradients/concat_grad/strided_slice_2/stack_1?
-gradients/concat_grad/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-gradients/concat_grad/strided_slice_2/stack_2?
%gradients/concat_grad/strided_slice_2StridedSlice&gradients/concat_grad/Shape_2:output:04gradients/concat_grad/strided_slice_2/stack:output:06gradients/concat_grad/strided_slice_2/stack_1:output:06gradients/concat_grad/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%gradients/concat_grad/strided_slice_2?
gradients/concat_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:?2
gradients/concat_grad/Shape_3?
gradients/concat_grad/add_3/yConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_grad/add_3/y?
gradients/concat_grad/add_3AddV2%gradients_concat_grad_add_concat_axis&gradients/concat_grad/add_3/y:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/add_3?
+gradients/concat_grad/strided_slice_3/stackPack%gradients_concat_grad_add_concat_axis*
N*
T0*
_output_shapes
:2-
+gradients/concat_grad/strided_slice_3/stack?
-gradients/concat_grad/strided_slice_3/stack_1Packgradients/concat_grad/add_3:z:0*
N*
T0*
_output_shapes
:2/
-gradients/concat_grad/strided_slice_3/stack_1?
-gradients/concat_grad/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-gradients/concat_grad/strided_slice_3/stack_2?
%gradients/concat_grad/strided_slice_3StridedSlice&gradients/concat_grad/Shape_3:output:04gradients/concat_grad/strided_slice_3/stack:output:06gradients/concat_grad/strided_slice_3/stack_1:output:06gradients/concat_grad/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%gradients/concat_grad/strided_slice_3?
gradients/concat_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:?2
gradients/concat_grad/Shape_4?
gradients/concat_grad/add_4/yConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_grad/add_4/y?
gradients/concat_grad/add_4AddV2%gradients_concat_grad_add_concat_axis&gradients/concat_grad/add_4/y:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/add_4?
+gradients/concat_grad/strided_slice_4/stackPack%gradients_concat_grad_add_concat_axis*
N*
T0*
_output_shapes
:2-
+gradients/concat_grad/strided_slice_4/stack?
-gradients/concat_grad/strided_slice_4/stack_1Packgradients/concat_grad/add_4:z:0*
N*
T0*
_output_shapes
:2/
-gradients/concat_grad/strided_slice_4/stack_1?
-gradients/concat_grad/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-gradients/concat_grad/strided_slice_4/stack_2?
%gradients/concat_grad/strided_slice_4StridedSlice&gradients/concat_grad/Shape_4:output:04gradients/concat_grad/strided_slice_4/stack:output:06gradients/concat_grad/strided_slice_4/stack_1:output:06gradients/concat_grad/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%gradients/concat_grad/strided_slice_4?
gradients/concat_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:?2
gradients/concat_grad/Shape_5?
gradients/concat_grad/add_5/yConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_grad/add_5/y?
gradients/concat_grad/add_5AddV2%gradients_concat_grad_add_concat_axis&gradients/concat_grad/add_5/y:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/add_5?
+gradients/concat_grad/strided_slice_5/stackPack%gradients_concat_grad_add_concat_axis*
N*
T0*
_output_shapes
:2-
+gradients/concat_grad/strided_slice_5/stack?
-gradients/concat_grad/strided_slice_5/stack_1Packgradients/concat_grad/add_5:z:0*
N*
T0*
_output_shapes
:2/
-gradients/concat_grad/strided_slice_5/stack_1?
-gradients/concat_grad/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-gradients/concat_grad/strided_slice_5/stack_2?
%gradients/concat_grad/strided_slice_5StridedSlice&gradients/concat_grad/Shape_5:output:04gradients/concat_grad/strided_slice_5/stack:output:06gradients/concat_grad/strided_slice_5/stack_1:output:06gradients/concat_grad/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%gradients/concat_grad/strided_slice_5?
gradients/concat_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape_6?
gradients/concat_grad/add_6/yConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_grad/add_6/y?
gradients/concat_grad/add_6AddV2%gradients_concat_grad_add_concat_axis&gradients/concat_grad/add_6/y:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/add_6?
+gradients/concat_grad/strided_slice_6/stackPack%gradients_concat_grad_add_concat_axis*
N*
T0*
_output_shapes
:2-
+gradients/concat_grad/strided_slice_6/stack?
-gradients/concat_grad/strided_slice_6/stack_1Packgradients/concat_grad/add_6:z:0*
N*
T0*
_output_shapes
:2/
-gradients/concat_grad/strided_slice_6/stack_1?
-gradients/concat_grad/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-gradients/concat_grad/strided_slice_6/stack_2?
%gradients/concat_grad/strided_slice_6StridedSlice&gradients/concat_grad/Shape_6:output:04gradients/concat_grad/strided_slice_6/stack:output:06gradients/concat_grad/strided_slice_6/stack_1:output:06gradients/concat_grad/strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%gradients/concat_grad/strided_slice_6?
gradients/concat_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape_7?
gradients/concat_grad/add_7/yConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_grad/add_7/y?
gradients/concat_grad/add_7AddV2%gradients_concat_grad_add_concat_axis&gradients/concat_grad/add_7/y:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/add_7?
+gradients/concat_grad/strided_slice_7/stackPack%gradients_concat_grad_add_concat_axis*
N*
T0*
_output_shapes
:2-
+gradients/concat_grad/strided_slice_7/stack?
-gradients/concat_grad/strided_slice_7/stack_1Packgradients/concat_grad/add_7:z:0*
N*
T0*
_output_shapes
:2/
-gradients/concat_grad/strided_slice_7/stack_1?
-gradients/concat_grad/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-gradients/concat_grad/strided_slice_7/stack_2?
%gradients/concat_grad/strided_slice_7StridedSlice&gradients/concat_grad/Shape_7:output:04gradients/concat_grad/strided_slice_7/stack:output:06gradients/concat_grad/strided_slice_7/stack_1:output:06gradients/concat_grad/strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%gradients/concat_grad/strided_slice_7?
gradients/concat_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape_8?
gradients/concat_grad/add_8/yConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_grad/add_8/y?
gradients/concat_grad/add_8AddV2%gradients_concat_grad_add_concat_axis&gradients/concat_grad/add_8/y:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/add_8?
+gradients/concat_grad/strided_slice_8/stackPack%gradients_concat_grad_add_concat_axis*
N*
T0*
_output_shapes
:2-
+gradients/concat_grad/strided_slice_8/stack?
-gradients/concat_grad/strided_slice_8/stack_1Packgradients/concat_grad/add_8:z:0*
N*
T0*
_output_shapes
:2/
-gradients/concat_grad/strided_slice_8/stack_1?
-gradients/concat_grad/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-gradients/concat_grad/strided_slice_8/stack_2?
%gradients/concat_grad/strided_slice_8StridedSlice&gradients/concat_grad/Shape_8:output:04gradients/concat_grad/strided_slice_8/stack:output:06gradients/concat_grad/strided_slice_8/stack_1:output:06gradients/concat_grad/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%gradients/concat_grad/strided_slice_8?
gradients/concat_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape_9?
gradients/concat_grad/add_9/yConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_grad/add_9/y?
gradients/concat_grad/add_9AddV2%gradients_concat_grad_add_concat_axis&gradients/concat_grad/add_9/y:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/add_9?
+gradients/concat_grad/strided_slice_9/stackPack%gradients_concat_grad_add_concat_axis*
N*
T0*
_output_shapes
:2-
+gradients/concat_grad/strided_slice_9/stack?
-gradients/concat_grad/strided_slice_9/stack_1Packgradients/concat_grad/add_9:z:0*
N*
T0*
_output_shapes
:2/
-gradients/concat_grad/strided_slice_9/stack_1?
-gradients/concat_grad/strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-gradients/concat_grad/strided_slice_9/stack_2?
%gradients/concat_grad/strided_slice_9StridedSlice&gradients/concat_grad/Shape_9:output:04gradients/concat_grad/strided_slice_9/stack:output:06gradients/concat_grad/strided_slice_9/stack_1:output:06gradients/concat_grad/strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%gradients/concat_grad/strided_slice_9?
gradients/concat_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB:2 
gradients/concat_grad/Shape_10?
gradients/concat_grad/add_10/yConst*
_output_shapes
: *
dtype0*
value	B :2 
gradients/concat_grad/add_10/y?
gradients/concat_grad/add_10AddV2%gradients_concat_grad_add_concat_axis'gradients/concat_grad/add_10/y:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/add_10?
,gradients/concat_grad/strided_slice_10/stackPack%gradients_concat_grad_add_concat_axis*
N*
T0*
_output_shapes
:2.
,gradients/concat_grad/strided_slice_10/stack?
.gradients/concat_grad/strided_slice_10/stack_1Pack gradients/concat_grad/add_10:z:0*
N*
T0*
_output_shapes
:20
.gradients/concat_grad/strided_slice_10/stack_1?
.gradients/concat_grad/strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.gradients/concat_grad/strided_slice_10/stack_2?
&gradients/concat_grad/strided_slice_10StridedSlice'gradients/concat_grad/Shape_10:output:05gradients/concat_grad/strided_slice_10/stack:output:07gradients/concat_grad/strided_slice_10/stack_1:output:07gradients/concat_grad/strided_slice_10/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2(
&gradients/concat_grad/strided_slice_10?
gradients/concat_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB:2 
gradients/concat_grad/Shape_11?
gradients/concat_grad/add_11/yConst*
_output_shapes
: *
dtype0*
value	B :2 
gradients/concat_grad/add_11/y?
gradients/concat_grad/add_11AddV2%gradients_concat_grad_add_concat_axis'gradients/concat_grad/add_11/y:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/add_11?
,gradients/concat_grad/strided_slice_11/stackPack%gradients_concat_grad_add_concat_axis*
N*
T0*
_output_shapes
:2.
,gradients/concat_grad/strided_slice_11/stack?
.gradients/concat_grad/strided_slice_11/stack_1Pack gradients/concat_grad/add_11:z:0*
N*
T0*
_output_shapes
:20
.gradients/concat_grad/strided_slice_11/stack_1?
.gradients/concat_grad/strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.gradients/concat_grad/strided_slice_11/stack_2?
&gradients/concat_grad/strided_slice_11StridedSlice'gradients/concat_grad/Shape_11:output:05gradients/concat_grad/strided_slice_11/stack:output:07gradients/concat_grad/strided_slice_11/stack_1:output:07gradients/concat_grad/strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2(
&gradients/concat_grad/strided_slice_11?
gradients/concat_grad/packedPack,gradients/concat_grad/strided_slice:output:0.gradients/concat_grad/strided_slice_1:output:0.gradients/concat_grad/strided_slice_2:output:0.gradients/concat_grad/strided_slice_3:output:0.gradients/concat_grad/strided_slice_4:output:0.gradients/concat_grad/strided_slice_5:output:0.gradients/concat_grad/strided_slice_6:output:0.gradients/concat_grad/strided_slice_7:output:0.gradients/concat_grad/strided_slice_8:output:0.gradients/concat_grad/strided_slice_9:output:0/gradients/concat_grad/strided_slice_10:output:0/gradients/concat_grad/strided_slice_11:output:0*
N*
T0*
_output_shapes
:2
gradients/concat_grad/packed?
%gradients/concat_grad/MLCConcatV2GradMLCConcatV2Grad:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0%gradients/concat_grad/packed:output:0%gradients_concat_grad_add_concat_axis$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0&gradients/concat_grad/Shape_2:output:0&gradients/concat_grad/Shape_3:output:0&gradients/concat_grad/Shape_4:output:0&gradients/concat_grad/Shape_5:output:0&gradients/concat_grad/Shape_6:output:0&gradients/concat_grad/Shape_7:output:0&gradients/concat_grad/Shape_8:output:0&gradients/concat_grad/Shape_9:output:0'gradients/concat_grad/Shape_10:output:0'gradients/concat_grad/Shape_11:output:0/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat*
N*
T0*

Tlen0*b
_output_shapesP
N:?:?:?:?:?:?::::::2'
%gradients/concat_grad/MLCConcatV2Grad?
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
gradients/Reshape_1_grad/Shape?
'gradients/Reshape_1_grad/MLCReshapeGradMLCReshapeGrad.gradients/concat_grad/MLCConcatV2Grad:output:0'gradients/Reshape_1_grad/Shape:output:01gradients_reshape_1_grad_mlcreshapegrad_reshape_1*
T0*
_output_shapes

:2)
'gradients/Reshape_1_grad/MLCReshapeGrad?
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
gradients/Reshape_2_grad/Shape?
'gradients/Reshape_2_grad/MLCReshapeGradMLCReshapeGrad.gradients/concat_grad/MLCConcatV2Grad:output:1'gradients/Reshape_2_grad/Shape:output:01gradients_reshape_2_grad_mlcreshapegrad_reshape_2*
T0*
_output_shapes

:2)
'gradients/Reshape_2_grad/MLCReshapeGrad?
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
gradients/Reshape_3_grad/Shape?
'gradients/Reshape_3_grad/MLCReshapeGradMLCReshapeGrad.gradients/concat_grad/MLCConcatV2Grad:output:2'gradients/Reshape_3_grad/Shape:output:01gradients_reshape_3_grad_mlcreshapegrad_reshape_3*
T0*
_output_shapes

:2)
'gradients/Reshape_3_grad/MLCReshapeGrad?
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
gradients/Reshape_4_grad/Shape?
'gradients/Reshape_4_grad/MLCReshapeGradMLCReshapeGrad.gradients/concat_grad/MLCConcatV2Grad:output:3'gradients/Reshape_4_grad/Shape:output:01gradients_reshape_4_grad_mlcreshapegrad_reshape_4*
T0*
_output_shapes

:2)
'gradients/Reshape_4_grad/MLCReshapeGrad?
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
gradients/Reshape_5_grad/Shape?
'gradients/Reshape_5_grad/MLCReshapeGradMLCReshapeGrad.gradients/concat_grad/MLCConcatV2Grad:output:4'gradients/Reshape_5_grad/Shape:output:01gradients_reshape_5_grad_mlcreshapegrad_reshape_5*
T0*
_output_shapes

:2)
'gradients/Reshape_5_grad/MLCReshapeGrad?
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
gradients/Reshape_6_grad/Shape?
'gradients/Reshape_6_grad/MLCReshapeGradMLCReshapeGrad.gradients/concat_grad/MLCConcatV2Grad:output:5'gradients/Reshape_6_grad/Shape:output:01gradients_reshape_6_grad_mlcreshapegrad_reshape_6*
T0*
_output_shapes

:2)
'gradients/Reshape_6_grad/MLCReshapeGrad?
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2 
gradients/Reshape_7_grad/Shape?
'gradients/Reshape_7_grad/MLCReshapeGradMLCReshapeGrad.gradients/concat_grad/MLCConcatV2Grad:output:6'gradients/Reshape_7_grad/Shape:output:01gradients_reshape_7_grad_mlcreshapegrad_reshape_7*
T0*
_output_shapes
:2)
'gradients/Reshape_7_grad/MLCReshapeGrad?
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2 
gradients/Reshape_8_grad/Shape?
'gradients/Reshape_8_grad/MLCReshapeGradMLCReshapeGrad.gradients/concat_grad/MLCConcatV2Grad:output:7'gradients/Reshape_8_grad/Shape:output:01gradients_reshape_8_grad_mlcreshapegrad_reshape_8*
T0*
_output_shapes
:2)
'gradients/Reshape_8_grad/MLCReshapeGrad?
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2 
gradients/Reshape_9_grad/Shape?
'gradients/Reshape_9_grad/MLCReshapeGradMLCReshapeGrad.gradients/concat_grad/MLCConcatV2Grad:output:8'gradients/Reshape_9_grad/Shape:output:01gradients_reshape_9_grad_mlcreshapegrad_reshape_9*
T0*
_output_shapes
:2)
'gradients/Reshape_9_grad/MLCReshapeGrad?
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2!
gradients/Reshape_10_grad/Shape?
(gradients/Reshape_10_grad/MLCReshapeGradMLCReshapeGrad.gradients/concat_grad/MLCConcatV2Grad:output:9(gradients/Reshape_10_grad/Shape:output:03gradients_reshape_10_grad_mlcreshapegrad_reshape_10*
T0*
_output_shapes
:2*
(gradients/Reshape_10_grad/MLCReshapeGrad?
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2!
gradients/Reshape_11_grad/Shape?
(gradients/Reshape_11_grad/MLCReshapeGradMLCReshapeGrad/gradients/concat_grad/MLCConcatV2Grad:output:10(gradients/Reshape_11_grad/Shape:output:03gradients_reshape_11_grad_mlcreshapegrad_reshape_11*
T0*
_output_shapes
:2*
(gradients/Reshape_11_grad/MLCReshapeGrad?
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2!
gradients/Reshape_12_grad/Shape?
(gradients/Reshape_12_grad/MLCReshapeGradMLCReshapeGrad/gradients/concat_grad/MLCConcatV2Grad:output:11(gradients/Reshape_12_grad/Shape:output:03gradients_reshape_12_grad_mlcreshapegrad_reshape_12*
T0*
_output_shapes
:2*
(gradients/Reshape_12_grad/MLCReshapeGrad?
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:2.
,gradients/transpose_1_grad/InvertPermutation?
+gradients/transpose_1_grad/MLCTransposeGradMLCTransposeGrad0gradients/Reshape_1_grad/MLCReshapeGrad:output:00gradients/transpose_1_grad/InvertPermutation:y:07gradients_transpose_1_grad_mlctransposegrad_transpose_1*
T0*
_output_shapes

:2-
+gradients/transpose_1_grad/MLCTransposeGrad?
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:2.
,gradients/transpose_2_grad/InvertPermutation?
+gradients/transpose_2_grad/MLCTransposeGradMLCTransposeGrad0gradients/Reshape_2_grad/MLCReshapeGrad:output:00gradients/transpose_2_grad/InvertPermutation:y:07gradients_transpose_2_grad_mlctransposegrad_transpose_2*
T0*
_output_shapes

:2-
+gradients/transpose_2_grad/MLCTransposeGrad?
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:2.
,gradients/transpose_3_grad/InvertPermutation?
+gradients/transpose_3_grad/MLCTransposeGradMLCTransposeGrad0gradients/Reshape_3_grad/MLCReshapeGrad:output:00gradients/transpose_3_grad/InvertPermutation:y:07gradients_transpose_3_grad_mlctransposegrad_transpose_3*
T0*
_output_shapes

:2-
+gradients/transpose_3_grad/MLCTransposeGrad?
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:2.
,gradients/transpose_4_grad/InvertPermutation?
+gradients/transpose_4_grad/MLCTransposeGradMLCTransposeGrad0gradients/Reshape_4_grad/MLCReshapeGrad:output:00gradients/transpose_4_grad/InvertPermutation:y:07gradients_transpose_4_grad_mlctransposegrad_transpose_4*
T0*
_output_shapes

:2-
+gradients/transpose_4_grad/MLCTransposeGrad?
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:2.
,gradients/transpose_5_grad/InvertPermutation?
+gradients/transpose_5_grad/MLCTransposeGradMLCTransposeGrad0gradients/Reshape_5_grad/MLCReshapeGrad:output:00gradients/transpose_5_grad/InvertPermutation:y:07gradients_transpose_5_grad_mlctransposegrad_transpose_5*
T0*
_output_shapes

:2-
+gradients/transpose_5_grad/MLCTransposeGrad?
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:2.
,gradients/transpose_6_grad/InvertPermutation?
+gradients/transpose_6_grad/MLCTransposeGradMLCTransposeGrad0gradients/Reshape_6_grad/MLCReshapeGrad:output:00gradients/transpose_6_grad/InvertPermutation:y:07gradients_transpose_6_grad_mlctransposegrad_transpose_6*
T0*
_output_shapes

:2-
+gradients/transpose_6_grad/MLCTransposeGrad?
gradients/split_2_grad/concatConcatV20gradients/Reshape_7_grad/MLCReshapeGrad:output:00gradients/Reshape_8_grad/MLCReshapeGrad:output:00gradients/Reshape_9_grad/MLCReshapeGrad:output:01gradients/Reshape_10_grad/MLCReshapeGrad:output:01gradients/Reshape_11_grad/MLCReshapeGrad:output:01gradients/Reshape_12_grad/MLCReshapeGrad:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes	
:?2
gradients/split_2_grad/concat?
!gradients/split_grad/MLCSplitGradMLCSplitGrad/gradients/transpose_1_grad/MLCTransposeGrad:y:0/gradients/transpose_2_grad/MLCTransposeGrad:y:0/gradients/transpose_3_grad/MLCTransposeGrad:y:01gradients_split_grad_mlcsplitgrad_split_split_dim'gradients_split_grad_mlcsplitgrad_split*
N*
T0*
_output_shapes

:Z2#
!gradients/split_grad/MLCSplitGrad?
#gradients/split_1_grad/MLCSplitGradMLCSplitGrad/gradients/transpose_4_grad/MLCTransposeGrad:y:0/gradients/transpose_5_grad/MLCTransposeGrad:y:0/gradients/transpose_6_grad/MLCTransposeGrad:y:05gradients_split_1_grad_mlcsplitgrad_split_1_split_dim+gradients_split_1_grad_mlcsplitgrad_split_1*
N*
T0*
_output_shapes

:Z2%
#gradients/split_1_grad/MLCSplitGrad?
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"   Z   2
gradients/Reshape_grad/Shape?
%gradients/Reshape_grad/MLCReshapeGradMLCReshapeGrad&gradients/split_2_grad/concat:output:0%gradients/Reshape_grad/Shape:output:0-gradients_reshape_grad_mlcreshapegrad_reshape*
T0*
_output_shapes

:Z2'
%gradients/Reshape_grad/MLCReshapeGrad?
IdentityIdentity-gradients/transpose_grad/MLCTransposeGrad:y:0*
T0*,
_output_shapes
:??????????2

Identity?

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0*
T0*'
_output_shapes
:?????????2

Identity_1y

Identity_2Identity*gradients/split_grad/MLCSplitGrad:output:0*
T0*
_output_shapes

:Z2

Identity_2{

Identity_3Identity,gradients/split_1_grad/MLCSplitGrad:output:0*
T0*
_output_shapes

:Z2

Identity_3}

Identity_4Identity.gradients/Reshape_grad/MLCReshapeGrad:output:0*
T0*
_output_shapes

:Z2

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*?
_input_shapes?
?:?????????:??????????:?????????: :??????????:?????????::??????????:?????????: ::??????????:?????????: :?::?????????: :?:?:?:?:?:?::::::::::::::::::: : :: ::?*<
api_implements*(gru_5f9dfa88-82d0-4c61-bd4c-33d58af38bad*
api_preferred_deviceGPU*@
forward_function_name'%__forward_gpu_gru_with_fallback_17763*
go_backwards( *

time_major( :- )
'
_output_shapes
:?????????:2.
,
_output_shapes
:??????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :2.
,
_output_shapes
:??????????:-)
'
_output_shapes
:?????????: 

_output_shapes
::2.
,
_output_shapes
:??????????:1-
+
_output_shapes
:?????????:	

_output_shapes
: :


_output_shapes
::2.
,
_output_shapes
:??????????:1-
+
_output_shapes
:?????????:

_output_shapes
: :!

_output_shapes	
:?: 

_output_shapes
::-)
'
_output_shapes
:?????????:

_output_shapes
: :!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::$ 

_output_shapes

::  

_output_shapes
::$! 

_output_shapes

:: "

_output_shapes
::$# 

_output_shapes

:: $

_output_shapes
::$% 

_output_shapes

:: &

_output_shapes
::$' 

_output_shapes

:: (

_output_shapes
::$) 

_output_shapes

::*

_output_shapes
: :+

_output_shapes
: :$, 

_output_shapes

::-

_output_shapes
: :$. 

_output_shapes

::!/

_output_shapes	
:?
?2
?
while_body_16275
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_matmul_kernel_0
while_biasadd_unstack_0%
!while_matmul_1_recurrent_kernel_0
while_biasadd_1_unstack_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_matmul_kernel
while_biasadd_unstack#
while_matmul_1_recurrent_kernel
while_biasadd_1_unstack?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
while/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while_matmul_kernel_0*
T0*'
_output_shapes
:?????????Z2
while/MatMul?
while/BiasAddBiasAddwhile/MatMul:product:0while_biasadd_unstack_0*
T0*'
_output_shapes
:?????????Z2
while/BiasAdd\
while/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/Constp
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
while/split/split_dim?
while/splitSplitwhile/split/split_dim:output:0while/BiasAdd:output:0*
T0*M
_output_shapes;
9:?????????:?????????:?????????*
	num_split2
while/split?
while/MatMul_1MatMulwhile_placeholder_2!while_matmul_1_recurrent_kernel_0*
T0*'
_output_shapes
:?????????Z2
while/MatMul_1?
while/BiasAdd_1BiasAddwhile/MatMul_1:product:0while_biasadd_1_unstack_0*
T0*'
_output_shapes
:?????????Z2
while/BiasAdd_1`
while/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
while/Const_1t
while/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
while/split_1/split_dim?
while/split_1Split while/split_1/split_dim:output:0while/BiasAdd_1:output:0*
T0*M
_output_shapes;
9:?????????:?????????:?????????*
	num_split2
while/split_1
	while/addAddV2while/split:output:0while/split_1:output:0*
T0*'
_output_shapes
:?????????2
	while/addj
while/SigmoidSigmoidwhile/add:z:0*
T0*'
_output_shapes
:?????????2
while/Sigmoid?
while/add_1AddV2while/split:output:1while/split_1:output:1*
T0*'
_output_shapes
:?????????2
while/add_1p
while/Sigmoid_1Sigmoidwhile/add_1:z:0*
T0*'
_output_shapes
:?????????2
while/Sigmoid_1|
	while/mulMulwhile/Sigmoid_1:y:0while/split_1:output:2*
T0*'
_output_shapes
:?????????2
	while/mulz
while/add_2AddV2while/split:output:2while/mul:z:0*
T0*'
_output_shapes
:?????????2
while/add_2c

while/TanhTanhwhile/add_2:z:0*
T0*'
_output_shapes
:?????????2

while/Tanh{
while/mul_1Mulwhile/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:?????????2
while/mul_1_
while/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
while/sub/xx
	while/subSubwhile/sub/x:output:0while/Sigmoid:y:0*
T0*'
_output_shapes
:?????????2
	while/subr
while/mul_2Mulwhile/sub:z:0while/Tanh:y:0*
T0*'
_output_shapes
:?????????2
while/mul_2w
while/add_3AddV2while/mul_1:z:0while/mul_2:z:0*
T0*'
_output_shapes
:?????????2
while/add_3?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/add_3:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem`
while/add_4/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_4/yo
while/add_4AddV2while_placeholderwhile/add_4/y:output:0*
T0*
_output_shapes
: 2
while/add_4`
while/add_5/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_5/yv
while/add_5AddV2while_while_loop_counterwhile/add_5/y:output:0*
T0*
_output_shapes
: 2
while/add_5^
while/IdentityIdentitywhile/add_5:z:0*
T0*
_output_shapes
: 2
while/Identityq
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: 2
while/Identity_1b
while/Identity_2Identitywhile/add_4:z:0*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3s
while/Identity_4Identitywhile/add_3:z:0*
T0*'
_output_shapes
:?????????2
while/Identity_4"4
while_biasadd_1_unstackwhile_biasadd_1_unstack_0"0
while_biasadd_unstackwhile_biasadd_unstack_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"D
while_matmul_1_recurrent_kernel!while_matmul_1_recurrent_kernel_0",
while_matmul_kernelwhile_matmul_kernel_0",
while_strided_slicewhile_strided_slice_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*R
_input_shapesA
?: : : : :?????????: : :Z:Z:Z:Z: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:Z: 

_output_shapes
:Z:$	 

_output_shapes

:Z: 


_output_shapes
:Z
ڮ
?
8__inference___backward_gpu_gru_with_fallback_15485_15703
placeholder
placeholder_1
placeholder_2
placeholder_3/
+gradients_strided_slice_grad_shape_cudnnrnnB
>gradients_strided_slice_grad_mlcstridedslicegrad_strided_sliceA
=gradients_transpose_7_grad_invertpermutation_transpose_7_perm;
7gradients_transpose_7_grad_mlctransposegrad_transpose_7)
%gradients_squeeze_grad_shape_cudnnrnn!
gradients_zeros_like_cudnnrnn#
gradients_zeros_like_1_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims=
9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c3
/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h)
%gradients_concat_grad_add_concat_axis5
1gradients_reshape_1_grad_mlcreshapegrad_reshape_15
1gradients_reshape_2_grad_mlcreshapegrad_reshape_25
1gradients_reshape_3_grad_mlcreshapegrad_reshape_35
1gradients_reshape_4_grad_mlcreshapegrad_reshape_45
1gradients_reshape_5_grad_mlcreshapegrad_reshape_55
1gradients_reshape_6_grad_mlcreshapegrad_reshape_65
1gradients_reshape_7_grad_mlcreshapegrad_reshape_75
1gradients_reshape_8_grad_mlcreshapegrad_reshape_85
1gradients_reshape_9_grad_mlcreshapegrad_reshape_97
3gradients_reshape_10_grad_mlcreshapegrad_reshape_107
3gradients_reshape_11_grad_mlcreshapegrad_reshape_117
3gradients_reshape_12_grad_mlcreshapegrad_reshape_12A
=gradients_transpose_1_grad_invertpermutation_transpose_1_perm;
7gradients_transpose_1_grad_mlctransposegrad_transpose_1A
=gradients_transpose_2_grad_invertpermutation_transpose_2_perm;
7gradients_transpose_2_grad_mlctransposegrad_transpose_2A
=gradients_transpose_3_grad_invertpermutation_transpose_3_perm;
7gradients_transpose_3_grad_mlctransposegrad_transpose_3A
=gradients_transpose_4_grad_invertpermutation_transpose_4_perm;
7gradients_transpose_4_grad_mlctransposegrad_transpose_4A
=gradients_transpose_5_grad_invertpermutation_transpose_5_perm;
7gradients_transpose_5_grad_mlctransposegrad_transpose_5A
=gradients_transpose_6_grad_invertpermutation_transpose_6_perm;
7gradients_transpose_6_grad_mlctransposegrad_transpose_63
/gradients_split_2_grad_concat_split_2_split_dim5
1gradients_split_grad_mlcsplitgrad_split_split_dim+
'gradients_split_grad_mlcsplitgrad_split9
5gradients_split_1_grad_mlcsplitgrad_split_1_split_dim/
+gradients_split_1_grad_mlcsplitgrad_split_11
-gradients_reshape_grad_mlcreshapegrad_reshape
identity

identity_1

identity_2

identity_3

identity_4?u
gradients/grad_ys_0Identityplaceholder*
T0*'
_output_shapes
:?????????2
gradients/grad_ys_0?
gradients/grad_ys_1Identityplaceholder_1*
T0*4
_output_shapes"
 :??????????????????2
gradients/grad_ys_1w
gradients/grad_ys_2Identityplaceholder_2*
T0*'
_output_shapes
:?????????2
gradients/grad_ys_2f
gradients/grad_ys_3Identityplaceholder_3*
T0*
_output_shapes
: 2
gradients/grad_ys_3?
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:2$
"gradients/strided_slice_grad/Shape?
6gradients/strided_slice_grad/MLCStridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*
valueB:
?????????28
6gradients/strided_slice_grad/MLCStridedSliceGrad/begin?
4gradients/strided_slice_grad/MLCStridedSliceGrad/endConst*
_output_shapes
:*
dtype0*
valueB: 26
4gradients/strided_slice_grad/MLCStridedSliceGrad/end?
8gradients/strided_slice_grad/MLCStridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*
valueB:2:
8gradients/strided_slice_grad/MLCStridedSliceGrad/strides?
0gradients/strided_slice_grad/MLCStridedSliceGradMLCStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0?gradients/strided_slice_grad/MLCStridedSliceGrad/begin:output:0=gradients/strided_slice_grad/MLCStridedSliceGrad/end:output:0Agradients/strided_slice_grad/MLCStridedSliceGrad/strides:output:0gradients/grad_ys_0:output:0>gradients_strided_slice_grad_mlcstridedslicegrad_strided_slice*
Index0*
T0*4
_output_shapes"
 :??????????????????*
shrink_axis_mask22
0gradients/strided_slice_grad/MLCStridedSliceGrad?
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:2.
,gradients/transpose_7_grad/InvertPermutation?
+gradients/transpose_7_grad/MLCTransposeGradMLCTransposeGradgradients/grad_ys_1:output:00gradients/transpose_7_grad/InvertPermutation:y:07gradients_transpose_7_grad_mlctransposegrad_transpose_7*
T0*4
_output_shapes"
 :??????????????????2-
+gradients/transpose_7_grad/MLCTransposeGrad?
gradients/Squeeze_grad/ShapeShape%gradients_squeeze_grad_shape_cudnnrnn*
T0*
_output_shapes
:2
gradients/Squeeze_grad/Shape?
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*+
_output_shapes
:?????????2 
gradients/Squeeze_grad/Reshape?
gradients/AddNAddN9gradients/strided_slice_grad/MLCStridedSliceGrad:output:0/gradients/transpose_7_grad/MLCTransposeGrad:y:0*
N*
T0*C
_class9
75loc:@gradients/strided_slice_grad/MLCStridedSliceGrad*4
_output_shapes"
 :??????????????????2
gradients/AddNy
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
: 2
gradients/zeros_like?
gradients/zeros_like_1	ZerosLikegradients_zeros_like_1_cudnnrnn*
T0*
_output_shapes
:2
gradients/zeros_like_1?
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat+gradients_strided_slice_grad_shape_cudnnrnn%gradients_squeeze_grad_shape_cudnnrnngradients_zeros_like_cudnnrnngradients/AddN:sum:0'gradients/Squeeze_grad/Reshape:output:0gradients/zeros_like:y:0gradients_zeros_like_1_cudnnrnn*
T0*T
_output_shapesB
@:??????????????????:?????????: :?*
rnn_modegru2*
(gradients/CudnnRNN_grad/CudnnRNNBackprop?
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:2,
*gradients/transpose_grad/InvertPermutation?
)gradients/transpose_grad/MLCTransposeGradMLCTransposeGrad9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:02gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose*
T0*4
_output_shapes"
 :??????????????????2+
)gradients/transpose_grad/MLCTransposeGrad?
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
:2!
gradients/ExpandDims_grad/Shape?
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*'
_output_shapes
:?????????2#
!gradients/ExpandDims_grad/Reshape?
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:?2
gradients/concat_grad/Shape|
gradients/concat_grad/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_grad/add/y?
gradients/concat_grad/addAddV2%gradients_concat_grad_add_concat_axis$gradients/concat_grad/add/y:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/add?
)gradients/concat_grad/strided_slice/stackPack%gradients_concat_grad_add_concat_axis*
N*
T0*
_output_shapes
:2+
)gradients/concat_grad/strided_slice/stack?
+gradients/concat_grad/strided_slice/stack_1Packgradients/concat_grad/add:z:0*
N*
T0*
_output_shapes
:2-
+gradients/concat_grad/strided_slice/stack_1?
+gradients/concat_grad/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+gradients/concat_grad/strided_slice/stack_2?
#gradients/concat_grad/strided_sliceStridedSlice$gradients/concat_grad/Shape:output:02gradients/concat_grad/strided_slice/stack:output:04gradients/concat_grad/strided_slice/stack_1:output:04gradients/concat_grad/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#gradients/concat_grad/strided_slice?
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:?2
gradients/concat_grad/Shape_1?
gradients/concat_grad/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_grad/add_1/y?
gradients/concat_grad/add_1AddV2%gradients_concat_grad_add_concat_axis&gradients/concat_grad/add_1/y:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/add_1?
+gradients/concat_grad/strided_slice_1/stackPack%gradients_concat_grad_add_concat_axis*
N*
T0*
_output_shapes
:2-
+gradients/concat_grad/strided_slice_1/stack?
-gradients/concat_grad/strided_slice_1/stack_1Packgradients/concat_grad/add_1:z:0*
N*
T0*
_output_shapes
:2/
-gradients/concat_grad/strided_slice_1/stack_1?
-gradients/concat_grad/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-gradients/concat_grad/strided_slice_1/stack_2?
%gradients/concat_grad/strided_slice_1StridedSlice&gradients/concat_grad/Shape_1:output:04gradients/concat_grad/strided_slice_1/stack:output:06gradients/concat_grad/strided_slice_1/stack_1:output:06gradients/concat_grad/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%gradients/concat_grad/strided_slice_1?
gradients/concat_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:?2
gradients/concat_grad/Shape_2?
gradients/concat_grad/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_grad/add_2/y?
gradients/concat_grad/add_2AddV2%gradients_concat_grad_add_concat_axis&gradients/concat_grad/add_2/y:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/add_2?
+gradients/concat_grad/strided_slice_2/stackPack%gradients_concat_grad_add_concat_axis*
N*
T0*
_output_shapes
:2-
+gradients/concat_grad/strided_slice_2/stack?
-gradients/concat_grad/strided_slice_2/stack_1Packgradients/concat_grad/add_2:z:0*
N*
T0*
_output_shapes
:2/
-gradients/concat_grad/strided_slice_2/stack_1?
-gradients/concat_grad/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-gradients/concat_grad/strided_slice_2/stack_2?
%gradients/concat_grad/strided_slice_2StridedSlice&gradients/concat_grad/Shape_2:output:04gradients/concat_grad/strided_slice_2/stack:output:06gradients/concat_grad/strided_slice_2/stack_1:output:06gradients/concat_grad/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%gradients/concat_grad/strided_slice_2?
gradients/concat_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:?2
gradients/concat_grad/Shape_3?
gradients/concat_grad/add_3/yConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_grad/add_3/y?
gradients/concat_grad/add_3AddV2%gradients_concat_grad_add_concat_axis&gradients/concat_grad/add_3/y:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/add_3?
+gradients/concat_grad/strided_slice_3/stackPack%gradients_concat_grad_add_concat_axis*
N*
T0*
_output_shapes
:2-
+gradients/concat_grad/strided_slice_3/stack?
-gradients/concat_grad/strided_slice_3/stack_1Packgradients/concat_grad/add_3:z:0*
N*
T0*
_output_shapes
:2/
-gradients/concat_grad/strided_slice_3/stack_1?
-gradients/concat_grad/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-gradients/concat_grad/strided_slice_3/stack_2?
%gradients/concat_grad/strided_slice_3StridedSlice&gradients/concat_grad/Shape_3:output:04gradients/concat_grad/strided_slice_3/stack:output:06gradients/concat_grad/strided_slice_3/stack_1:output:06gradients/concat_grad/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%gradients/concat_grad/strided_slice_3?
gradients/concat_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:?2
gradients/concat_grad/Shape_4?
gradients/concat_grad/add_4/yConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_grad/add_4/y?
gradients/concat_grad/add_4AddV2%gradients_concat_grad_add_concat_axis&gradients/concat_grad/add_4/y:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/add_4?
+gradients/concat_grad/strided_slice_4/stackPack%gradients_concat_grad_add_concat_axis*
N*
T0*
_output_shapes
:2-
+gradients/concat_grad/strided_slice_4/stack?
-gradients/concat_grad/strided_slice_4/stack_1Packgradients/concat_grad/add_4:z:0*
N*
T0*
_output_shapes
:2/
-gradients/concat_grad/strided_slice_4/stack_1?
-gradients/concat_grad/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-gradients/concat_grad/strided_slice_4/stack_2?
%gradients/concat_grad/strided_slice_4StridedSlice&gradients/concat_grad/Shape_4:output:04gradients/concat_grad/strided_slice_4/stack:output:06gradients/concat_grad/strided_slice_4/stack_1:output:06gradients/concat_grad/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%gradients/concat_grad/strided_slice_4?
gradients/concat_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:?2
gradients/concat_grad/Shape_5?
gradients/concat_grad/add_5/yConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_grad/add_5/y?
gradients/concat_grad/add_5AddV2%gradients_concat_grad_add_concat_axis&gradients/concat_grad/add_5/y:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/add_5?
+gradients/concat_grad/strided_slice_5/stackPack%gradients_concat_grad_add_concat_axis*
N*
T0*
_output_shapes
:2-
+gradients/concat_grad/strided_slice_5/stack?
-gradients/concat_grad/strided_slice_5/stack_1Packgradients/concat_grad/add_5:z:0*
N*
T0*
_output_shapes
:2/
-gradients/concat_grad/strided_slice_5/stack_1?
-gradients/concat_grad/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-gradients/concat_grad/strided_slice_5/stack_2?
%gradients/concat_grad/strided_slice_5StridedSlice&gradients/concat_grad/Shape_5:output:04gradients/concat_grad/strided_slice_5/stack:output:06gradients/concat_grad/strided_slice_5/stack_1:output:06gradients/concat_grad/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%gradients/concat_grad/strided_slice_5?
gradients/concat_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape_6?
gradients/concat_grad/add_6/yConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_grad/add_6/y?
gradients/concat_grad/add_6AddV2%gradients_concat_grad_add_concat_axis&gradients/concat_grad/add_6/y:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/add_6?
+gradients/concat_grad/strided_slice_6/stackPack%gradients_concat_grad_add_concat_axis*
N*
T0*
_output_shapes
:2-
+gradients/concat_grad/strided_slice_6/stack?
-gradients/concat_grad/strided_slice_6/stack_1Packgradients/concat_grad/add_6:z:0*
N*
T0*
_output_shapes
:2/
-gradients/concat_grad/strided_slice_6/stack_1?
-gradients/concat_grad/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-gradients/concat_grad/strided_slice_6/stack_2?
%gradients/concat_grad/strided_slice_6StridedSlice&gradients/concat_grad/Shape_6:output:04gradients/concat_grad/strided_slice_6/stack:output:06gradients/concat_grad/strided_slice_6/stack_1:output:06gradients/concat_grad/strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%gradients/concat_grad/strided_slice_6?
gradients/concat_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape_7?
gradients/concat_grad/add_7/yConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_grad/add_7/y?
gradients/concat_grad/add_7AddV2%gradients_concat_grad_add_concat_axis&gradients/concat_grad/add_7/y:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/add_7?
+gradients/concat_grad/strided_slice_7/stackPack%gradients_concat_grad_add_concat_axis*
N*
T0*
_output_shapes
:2-
+gradients/concat_grad/strided_slice_7/stack?
-gradients/concat_grad/strided_slice_7/stack_1Packgradients/concat_grad/add_7:z:0*
N*
T0*
_output_shapes
:2/
-gradients/concat_grad/strided_slice_7/stack_1?
-gradients/concat_grad/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-gradients/concat_grad/strided_slice_7/stack_2?
%gradients/concat_grad/strided_slice_7StridedSlice&gradients/concat_grad/Shape_7:output:04gradients/concat_grad/strided_slice_7/stack:output:06gradients/concat_grad/strided_slice_7/stack_1:output:06gradients/concat_grad/strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%gradients/concat_grad/strided_slice_7?
gradients/concat_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape_8?
gradients/concat_grad/add_8/yConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_grad/add_8/y?
gradients/concat_grad/add_8AddV2%gradients_concat_grad_add_concat_axis&gradients/concat_grad/add_8/y:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/add_8?
+gradients/concat_grad/strided_slice_8/stackPack%gradients_concat_grad_add_concat_axis*
N*
T0*
_output_shapes
:2-
+gradients/concat_grad/strided_slice_8/stack?
-gradients/concat_grad/strided_slice_8/stack_1Packgradients/concat_grad/add_8:z:0*
N*
T0*
_output_shapes
:2/
-gradients/concat_grad/strided_slice_8/stack_1?
-gradients/concat_grad/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-gradients/concat_grad/strided_slice_8/stack_2?
%gradients/concat_grad/strided_slice_8StridedSlice&gradients/concat_grad/Shape_8:output:04gradients/concat_grad/strided_slice_8/stack:output:06gradients/concat_grad/strided_slice_8/stack_1:output:06gradients/concat_grad/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%gradients/concat_grad/strided_slice_8?
gradients/concat_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape_9?
gradients/concat_grad/add_9/yConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_grad/add_9/y?
gradients/concat_grad/add_9AddV2%gradients_concat_grad_add_concat_axis&gradients/concat_grad/add_9/y:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/add_9?
+gradients/concat_grad/strided_slice_9/stackPack%gradients_concat_grad_add_concat_axis*
N*
T0*
_output_shapes
:2-
+gradients/concat_grad/strided_slice_9/stack?
-gradients/concat_grad/strided_slice_9/stack_1Packgradients/concat_grad/add_9:z:0*
N*
T0*
_output_shapes
:2/
-gradients/concat_grad/strided_slice_9/stack_1?
-gradients/concat_grad/strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-gradients/concat_grad/strided_slice_9/stack_2?
%gradients/concat_grad/strided_slice_9StridedSlice&gradients/concat_grad/Shape_9:output:04gradients/concat_grad/strided_slice_9/stack:output:06gradients/concat_grad/strided_slice_9/stack_1:output:06gradients/concat_grad/strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%gradients/concat_grad/strided_slice_9?
gradients/concat_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB:2 
gradients/concat_grad/Shape_10?
gradients/concat_grad/add_10/yConst*
_output_shapes
: *
dtype0*
value	B :2 
gradients/concat_grad/add_10/y?
gradients/concat_grad/add_10AddV2%gradients_concat_grad_add_concat_axis'gradients/concat_grad/add_10/y:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/add_10?
,gradients/concat_grad/strided_slice_10/stackPack%gradients_concat_grad_add_concat_axis*
N*
T0*
_output_shapes
:2.
,gradients/concat_grad/strided_slice_10/stack?
.gradients/concat_grad/strided_slice_10/stack_1Pack gradients/concat_grad/add_10:z:0*
N*
T0*
_output_shapes
:20
.gradients/concat_grad/strided_slice_10/stack_1?
.gradients/concat_grad/strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.gradients/concat_grad/strided_slice_10/stack_2?
&gradients/concat_grad/strided_slice_10StridedSlice'gradients/concat_grad/Shape_10:output:05gradients/concat_grad/strided_slice_10/stack:output:07gradients/concat_grad/strided_slice_10/stack_1:output:07gradients/concat_grad/strided_slice_10/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2(
&gradients/concat_grad/strided_slice_10?
gradients/concat_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB:2 
gradients/concat_grad/Shape_11?
gradients/concat_grad/add_11/yConst*
_output_shapes
: *
dtype0*
value	B :2 
gradients/concat_grad/add_11/y?
gradients/concat_grad/add_11AddV2%gradients_concat_grad_add_concat_axis'gradients/concat_grad/add_11/y:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/add_11?
,gradients/concat_grad/strided_slice_11/stackPack%gradients_concat_grad_add_concat_axis*
N*
T0*
_output_shapes
:2.
,gradients/concat_grad/strided_slice_11/stack?
.gradients/concat_grad/strided_slice_11/stack_1Pack gradients/concat_grad/add_11:z:0*
N*
T0*
_output_shapes
:20
.gradients/concat_grad/strided_slice_11/stack_1?
.gradients/concat_grad/strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.gradients/concat_grad/strided_slice_11/stack_2?
&gradients/concat_grad/strided_slice_11StridedSlice'gradients/concat_grad/Shape_11:output:05gradients/concat_grad/strided_slice_11/stack:output:07gradients/concat_grad/strided_slice_11/stack_1:output:07gradients/concat_grad/strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2(
&gradients/concat_grad/strided_slice_11?
gradients/concat_grad/packedPack,gradients/concat_grad/strided_slice:output:0.gradients/concat_grad/strided_slice_1:output:0.gradients/concat_grad/strided_slice_2:output:0.gradients/concat_grad/strided_slice_3:output:0.gradients/concat_grad/strided_slice_4:output:0.gradients/concat_grad/strided_slice_5:output:0.gradients/concat_grad/strided_slice_6:output:0.gradients/concat_grad/strided_slice_7:output:0.gradients/concat_grad/strided_slice_8:output:0.gradients/concat_grad/strided_slice_9:output:0/gradients/concat_grad/strided_slice_10:output:0/gradients/concat_grad/strided_slice_11:output:0*
N*
T0*
_output_shapes
:2
gradients/concat_grad/packed?
%gradients/concat_grad/MLCConcatV2GradMLCConcatV2Grad:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0%gradients/concat_grad/packed:output:0%gradients_concat_grad_add_concat_axis$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0&gradients/concat_grad/Shape_2:output:0&gradients/concat_grad/Shape_3:output:0&gradients/concat_grad/Shape_4:output:0&gradients/concat_grad/Shape_5:output:0&gradients/concat_grad/Shape_6:output:0&gradients/concat_grad/Shape_7:output:0&gradients/concat_grad/Shape_8:output:0&gradients/concat_grad/Shape_9:output:0'gradients/concat_grad/Shape_10:output:0'gradients/concat_grad/Shape_11:output:0/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat*
N*
T0*

Tlen0*b
_output_shapesP
N:?:?:?:?:?:?::::::2'
%gradients/concat_grad/MLCConcatV2Grad?
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
gradients/Reshape_1_grad/Shape?
'gradients/Reshape_1_grad/MLCReshapeGradMLCReshapeGrad.gradients/concat_grad/MLCConcatV2Grad:output:0'gradients/Reshape_1_grad/Shape:output:01gradients_reshape_1_grad_mlcreshapegrad_reshape_1*
T0*
_output_shapes

:2)
'gradients/Reshape_1_grad/MLCReshapeGrad?
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
gradients/Reshape_2_grad/Shape?
'gradients/Reshape_2_grad/MLCReshapeGradMLCReshapeGrad.gradients/concat_grad/MLCConcatV2Grad:output:1'gradients/Reshape_2_grad/Shape:output:01gradients_reshape_2_grad_mlcreshapegrad_reshape_2*
T0*
_output_shapes

:2)
'gradients/Reshape_2_grad/MLCReshapeGrad?
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
gradients/Reshape_3_grad/Shape?
'gradients/Reshape_3_grad/MLCReshapeGradMLCReshapeGrad.gradients/concat_grad/MLCConcatV2Grad:output:2'gradients/Reshape_3_grad/Shape:output:01gradients_reshape_3_grad_mlcreshapegrad_reshape_3*
T0*
_output_shapes

:2)
'gradients/Reshape_3_grad/MLCReshapeGrad?
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
gradients/Reshape_4_grad/Shape?
'gradients/Reshape_4_grad/MLCReshapeGradMLCReshapeGrad.gradients/concat_grad/MLCConcatV2Grad:output:3'gradients/Reshape_4_grad/Shape:output:01gradients_reshape_4_grad_mlcreshapegrad_reshape_4*
T0*
_output_shapes

:2)
'gradients/Reshape_4_grad/MLCReshapeGrad?
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
gradients/Reshape_5_grad/Shape?
'gradients/Reshape_5_grad/MLCReshapeGradMLCReshapeGrad.gradients/concat_grad/MLCConcatV2Grad:output:4'gradients/Reshape_5_grad/Shape:output:01gradients_reshape_5_grad_mlcreshapegrad_reshape_5*
T0*
_output_shapes

:2)
'gradients/Reshape_5_grad/MLCReshapeGrad?
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
gradients/Reshape_6_grad/Shape?
'gradients/Reshape_6_grad/MLCReshapeGradMLCReshapeGrad.gradients/concat_grad/MLCConcatV2Grad:output:5'gradients/Reshape_6_grad/Shape:output:01gradients_reshape_6_grad_mlcreshapegrad_reshape_6*
T0*
_output_shapes

:2)
'gradients/Reshape_6_grad/MLCReshapeGrad?
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2 
gradients/Reshape_7_grad/Shape?
'gradients/Reshape_7_grad/MLCReshapeGradMLCReshapeGrad.gradients/concat_grad/MLCConcatV2Grad:output:6'gradients/Reshape_7_grad/Shape:output:01gradients_reshape_7_grad_mlcreshapegrad_reshape_7*
T0*
_output_shapes
:2)
'gradients/Reshape_7_grad/MLCReshapeGrad?
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2 
gradients/Reshape_8_grad/Shape?
'gradients/Reshape_8_grad/MLCReshapeGradMLCReshapeGrad.gradients/concat_grad/MLCConcatV2Grad:output:7'gradients/Reshape_8_grad/Shape:output:01gradients_reshape_8_grad_mlcreshapegrad_reshape_8*
T0*
_output_shapes
:2)
'gradients/Reshape_8_grad/MLCReshapeGrad?
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2 
gradients/Reshape_9_grad/Shape?
'gradients/Reshape_9_grad/MLCReshapeGradMLCReshapeGrad.gradients/concat_grad/MLCConcatV2Grad:output:8'gradients/Reshape_9_grad/Shape:output:01gradients_reshape_9_grad_mlcreshapegrad_reshape_9*
T0*
_output_shapes
:2)
'gradients/Reshape_9_grad/MLCReshapeGrad?
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2!
gradients/Reshape_10_grad/Shape?
(gradients/Reshape_10_grad/MLCReshapeGradMLCReshapeGrad.gradients/concat_grad/MLCConcatV2Grad:output:9(gradients/Reshape_10_grad/Shape:output:03gradients_reshape_10_grad_mlcreshapegrad_reshape_10*
T0*
_output_shapes
:2*
(gradients/Reshape_10_grad/MLCReshapeGrad?
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2!
gradients/Reshape_11_grad/Shape?
(gradients/Reshape_11_grad/MLCReshapeGradMLCReshapeGrad/gradients/concat_grad/MLCConcatV2Grad:output:10(gradients/Reshape_11_grad/Shape:output:03gradients_reshape_11_grad_mlcreshapegrad_reshape_11*
T0*
_output_shapes
:2*
(gradients/Reshape_11_grad/MLCReshapeGrad?
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2!
gradients/Reshape_12_grad/Shape?
(gradients/Reshape_12_grad/MLCReshapeGradMLCReshapeGrad/gradients/concat_grad/MLCConcatV2Grad:output:11(gradients/Reshape_12_grad/Shape:output:03gradients_reshape_12_grad_mlcreshapegrad_reshape_12*
T0*
_output_shapes
:2*
(gradients/Reshape_12_grad/MLCReshapeGrad?
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:2.
,gradients/transpose_1_grad/InvertPermutation?
+gradients/transpose_1_grad/MLCTransposeGradMLCTransposeGrad0gradients/Reshape_1_grad/MLCReshapeGrad:output:00gradients/transpose_1_grad/InvertPermutation:y:07gradients_transpose_1_grad_mlctransposegrad_transpose_1*
T0*
_output_shapes

:2-
+gradients/transpose_1_grad/MLCTransposeGrad?
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:2.
,gradients/transpose_2_grad/InvertPermutation?
+gradients/transpose_2_grad/MLCTransposeGradMLCTransposeGrad0gradients/Reshape_2_grad/MLCReshapeGrad:output:00gradients/transpose_2_grad/InvertPermutation:y:07gradients_transpose_2_grad_mlctransposegrad_transpose_2*
T0*
_output_shapes

:2-
+gradients/transpose_2_grad/MLCTransposeGrad?
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:2.
,gradients/transpose_3_grad/InvertPermutation?
+gradients/transpose_3_grad/MLCTransposeGradMLCTransposeGrad0gradients/Reshape_3_grad/MLCReshapeGrad:output:00gradients/transpose_3_grad/InvertPermutation:y:07gradients_transpose_3_grad_mlctransposegrad_transpose_3*
T0*
_output_shapes

:2-
+gradients/transpose_3_grad/MLCTransposeGrad?
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:2.
,gradients/transpose_4_grad/InvertPermutation?
+gradients/transpose_4_grad/MLCTransposeGradMLCTransposeGrad0gradients/Reshape_4_grad/MLCReshapeGrad:output:00gradients/transpose_4_grad/InvertPermutation:y:07gradients_transpose_4_grad_mlctransposegrad_transpose_4*
T0*
_output_shapes

:2-
+gradients/transpose_4_grad/MLCTransposeGrad?
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:2.
,gradients/transpose_5_grad/InvertPermutation?
+gradients/transpose_5_grad/MLCTransposeGradMLCTransposeGrad0gradients/Reshape_5_grad/MLCReshapeGrad:output:00gradients/transpose_5_grad/InvertPermutation:y:07gradients_transpose_5_grad_mlctransposegrad_transpose_5*
T0*
_output_shapes

:2-
+gradients/transpose_5_grad/MLCTransposeGrad?
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:2.
,gradients/transpose_6_grad/InvertPermutation?
+gradients/transpose_6_grad/MLCTransposeGradMLCTransposeGrad0gradients/Reshape_6_grad/MLCReshapeGrad:output:00gradients/transpose_6_grad/InvertPermutation:y:07gradients_transpose_6_grad_mlctransposegrad_transpose_6*
T0*
_output_shapes

:2-
+gradients/transpose_6_grad/MLCTransposeGrad?
gradients/split_2_grad/concatConcatV20gradients/Reshape_7_grad/MLCReshapeGrad:output:00gradients/Reshape_8_grad/MLCReshapeGrad:output:00gradients/Reshape_9_grad/MLCReshapeGrad:output:01gradients/Reshape_10_grad/MLCReshapeGrad:output:01gradients/Reshape_11_grad/MLCReshapeGrad:output:01gradients/Reshape_12_grad/MLCReshapeGrad:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes	
:?2
gradients/split_2_grad/concat?
!gradients/split_grad/MLCSplitGradMLCSplitGrad/gradients/transpose_1_grad/MLCTransposeGrad:y:0/gradients/transpose_2_grad/MLCTransposeGrad:y:0/gradients/transpose_3_grad/MLCTransposeGrad:y:01gradients_split_grad_mlcsplitgrad_split_split_dim'gradients_split_grad_mlcsplitgrad_split*
N*
T0*
_output_shapes

:Z2#
!gradients/split_grad/MLCSplitGrad?
#gradients/split_1_grad/MLCSplitGradMLCSplitGrad/gradients/transpose_4_grad/MLCTransposeGrad:y:0/gradients/transpose_5_grad/MLCTransposeGrad:y:0/gradients/transpose_6_grad/MLCTransposeGrad:y:05gradients_split_1_grad_mlcsplitgrad_split_1_split_dim+gradients_split_1_grad_mlcsplitgrad_split_1*
N*
T0*
_output_shapes

:Z2%
#gradients/split_1_grad/MLCSplitGrad?
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"   Z   2
gradients/Reshape_grad/Shape?
%gradients/Reshape_grad/MLCReshapeGradMLCReshapeGrad&gradients/split_2_grad/concat:output:0%gradients/Reshape_grad/Shape:output:0-gradients_reshape_grad_mlcreshapegrad_reshape*
T0*
_output_shapes

:Z2'
%gradients/Reshape_grad/MLCReshapeGrad?
IdentityIdentity-gradients/transpose_grad/MLCTransposeGrad:y:0*
T0*4
_output_shapes"
 :??????????????????2

Identity?

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0*
T0*'
_output_shapes
:?????????2

Identity_1y

Identity_2Identity*gradients/split_grad/MLCSplitGrad:output:0*
T0*
_output_shapes

:Z2

Identity_2{

Identity_3Identity,gradients/split_1_grad/MLCSplitGrad:output:0*
T0*
_output_shapes

:Z2

Identity_3}

Identity_4Identity.gradients/Reshape_grad/MLCReshapeGrad:output:0*
T0*
_output_shapes

:Z2

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*?
_input_shapes?
?:?????????:??????????????????:?????????: :??????????????????:?????????::??????????????????:?????????: ::??????????????????:?????????: :?::?????????: :?:?:?:?:?:?::::::::::::::::::: : :: ::?*<
api_implements*(gru_85420d5a-5eb6-4eb6-bfb4-ef0c67d578e4*
api_preferred_deviceGPU*@
forward_function_name'%__forward_gpu_gru_with_fallback_15702*
go_backwards( *

time_major( :- )
'
_output_shapes
:?????????::6
4
_output_shapes"
 :??????????????????:-)
'
_output_shapes
:?????????:

_output_shapes
: ::6
4
_output_shapes"
 :??????????????????:-)
'
_output_shapes
:?????????: 

_output_shapes
:::6
4
_output_shapes"
 :??????????????????:1-
+
_output_shapes
:?????????:	

_output_shapes
: :


_output_shapes
:::6
4
_output_shapes"
 :??????????????????:1-
+
_output_shapes
:?????????:

_output_shapes
: :!

_output_shapes	
:?: 

_output_shapes
::-)
'
_output_shapes
:?????????:

_output_shapes
: :!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::$ 

_output_shapes

::  

_output_shapes
::$! 

_output_shapes

:: "

_output_shapes
::$# 

_output_shapes

:: $

_output_shapes
::$% 

_output_shapes

:: &

_output_shapes
::$' 

_output_shapes

:: (

_output_shapes
::$) 

_output_shapes

::*

_output_shapes
: :+

_output_shapes
: :$, 

_output_shapes

::-

_output_shapes
: :$. 

_output_shapes

::!/

_output_shapes	
:?
?M
?
%__forward_gpu_gru_with_fallback_15702

inputs
init_h_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3
cudnnrnn
strided_slice
transpose_7_perm
transpose_7

cudnnrnn_0

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
cudnnrnn_input_c

concat
transpose_perm

init_h
concat_axis
	reshape_1
	reshape_2
	reshape_3
	reshape_4
	reshape_5
	reshape_6
	reshape_7
	reshape_8
	reshape_9

reshape_10

reshape_11

reshape_12
transpose_1_perm
transpose_1
transpose_2_perm
transpose_2
transpose_3_perm
transpose_3
transpose_4_perm
transpose_4
transpose_5_perm
transpose_5
transpose_6_perm
transpose_6
split_2_split_dim
split_split_dim	
split
split_1_split_dim
split_1
reshape?u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permO
transpose_0	Transposeinputstranspose/perm:output:0*
T02
	transposeb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims/dim

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2

ExpandDimsP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dimU
split_0Splitsplit/split_dim:output:0kernel*
T0*
	num_split2
splitT
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dime
	split_1_0Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*
	num_split2	
split_1q
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
Reshape/shapea
ReshapeReshapebiasReshape/shape:output:0*
T0*
_output_shapes	
:?2	
ReshapeT
Const_2Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_2h
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_2/split_dim?
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*8
_output_shapes&
$::::::*
	num_split2	
split_2e
Const_3Const*
_output_shapes
:*
dtype0*
valueB:
?????????2	
Const_3u
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_1/perm_
transpose_1_0	Transposesplit_0:output:0transpose_1/perm:output:0*
T02
transpose_1l
	Reshape_1Reshapetranspose_1_0:y:0Const_3:output:0*
T0*
_output_shapes	
:?2
	Reshape_1u
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_2/perm_
transpose_2_0	Transposesplit_0:output:1transpose_2/perm:output:0*
T02
transpose_2l
	Reshape_2Reshapetranspose_2_0:y:0Const_3:output:0*
T0*
_output_shapes	
:?2
	Reshape_2u
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_3/perm_
transpose_3_0	Transposesplit_0:output:2transpose_3/perm:output:0*
T02
transpose_3l
	Reshape_3Reshapetranspose_3_0:y:0Const_3:output:0*
T0*
_output_shapes	
:?2
	Reshape_3u
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_4/perma
transpose_4_0	Transposesplit_1_0:output:0transpose_4/perm:output:0*
T02
transpose_4l
	Reshape_4Reshapetranspose_4_0:y:0Const_3:output:0*
T0*
_output_shapes	
:?2
	Reshape_4u
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_5/perma
transpose_5_0	Transposesplit_1_0:output:1transpose_5/perm:output:0*
T02
transpose_5l
	Reshape_5Reshapetranspose_5_0:y:0Const_3:output:0*
T0*
_output_shapes	
:?2
	Reshape_5u
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_6/perma
transpose_6_0	Transposesplit_1_0:output:2transpose_6/perm:output:0*
T02
transpose_6l
	Reshape_6Reshapetranspose_6_0:y:0Const_3:output:0*
T0*
_output_shapes	
:?2
	Reshape_6j
	Reshape_7Reshapesplit_2:output:0Const_3:output:0*
T0*
_output_shapes
:2
	Reshape_7j
	Reshape_8Reshapesplit_2:output:1Const_3:output:0*
T0*
_output_shapes
:2
	Reshape_8j
	Reshape_9Reshapesplit_2:output:2Const_3:output:0*
T0*
_output_shapes
:2
	Reshape_9l

Reshape_10Reshapesplit_2:output:3Const_3:output:0*
T0*
_output_shapes
:2

Reshape_10l

Reshape_11Reshapesplit_2:output:4Const_3:output:0*
T0*
_output_shapes
:2

Reshape_11l

Reshape_12Reshapesplit_2:output:5Const_3:output:0*
T0*
_output_shapes
:2

Reshape_12\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat/axis?
concat_0ConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
N*
T02
concati
CudnnRNN/input_cConst*
_output_shapes
: *
dtype0*
valueB
 *    2
CudnnRNN/input_c?
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat_0:output:0*
T0*Q
_output_shapes?
=:??????????????????:?????????: :*
rnn_modegru2

CudnnRNN}
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_slice_0StridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
shrink_axis_mask2
strided_slicey
transpose_7/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_7/perm`
transpose_7_0	TransposeCudnnRNN:output:0transpose_7/perm:output:0*
T02
transpose_7{
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:?????????*
squeeze_dims
 2	
Squeezef
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @2	
runtimel
IdentityIdentitystrided_slice_0:output:0*
T0*'
_output_shapes
:?????????2

Identityv

Identity_1Identitytranspose_7_0:y:0*
T0*4
_output_shapes"
 :??????????????????2

Identity_1h

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:?????????2

Identity_2W

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: 2

Identity_3"
concatconcat_0:output:0"#
concat_axisconcat/axis:output:0"
cudnnrnnCudnnRNN:output:0"!

cudnnrnn_0CudnnRNN:output_h:0"!

cudnnrnn_1CudnnRNN:output_c:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"-
cudnnrnn_input_cCudnnRNN/input_c:output:0"!

expanddimsExpandDims:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"
init_hinit_h_0"
reshapeReshape:output:0"
	reshape_1Reshape_1:output:0"!

reshape_10Reshape_10:output:0"!

reshape_11Reshape_11:output:0"!

reshape_12Reshape_12:output:0"
	reshape_2Reshape_2:output:0"
	reshape_3Reshape_3:output:0"
	reshape_4Reshape_4:output:0"
	reshape_5Reshape_5:output:0"
	reshape_6Reshape_6:output:0"
	reshape_7Reshape_7:output:0"
	reshape_8Reshape_8:output:0"
	reshape_9Reshape_9:output:0"
splitsplit_0:output:0"
split_1split_1_0:output:0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0")
strided_slicestrided_slice_0:output:0"
	transposetranspose_0:y:0" 
transpose_1transpose_1_0:y:0"-
transpose_1_permtranspose_1/perm:output:0" 
transpose_2transpose_2_0:y:0"-
transpose_2_permtranspose_2/perm:output:0" 
transpose_3transpose_3_0:y:0"-
transpose_3_permtranspose_3/perm:output:0" 
transpose_4transpose_4_0:y:0"-
transpose_4_permtranspose_4/perm:output:0" 
transpose_5transpose_5_0:y:0"-
transpose_5_permtranspose_5/perm:output:0" 
transpose_6transpose_6_0:y:0"-
transpose_6_permtranspose_6/perm:output:0" 
transpose_7transpose_7_0:y:0"-
transpose_7_permtranspose_7/perm:output:0")
transpose_permtranspose/perm:output:0*d
_input_shapesS
Q:??????????????????:?????????:Z:Z:Z*<
api_implements*(gru_85420d5a-5eb6-4eb6-bfb4-ef0c67d578e4*
api_preferred_deviceGPU*T
backward_function_name:8__inference___backward_gpu_gru_with_fallback_15485_15703*
go_backwards( *

time_major( :\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinit_h:FB

_output_shapes

:Z
 
_user_specified_namekernel:PL

_output_shapes

:Z
*
_user_specified_namerecurrent_kernel:D@

_output_shapes

:Z

_user_specified_namebias
?E
?
__inference_standard_gru_15405

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3X
unstackUnpackbias*
T0* 
_output_shapes
:Z:Z*	
num2	
unstacku
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????2
	transposeK
ShapeShapetranspose:y:0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_1n
MatMulMatMulstrided_slice_1:output:0kernel*
T0*'
_output_shapes
:?????????Z2
MatMuls
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*'
_output_shapes
:?????????Z2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:?????????:?????????:?????????*
	num_split2
splitj
MatMul_1MatMulinit_hrecurrent_kernel*
T0*'
_output_shapes
:?????????Z2

MatMul_1y
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*'
_output_shapes
:?????????Z2
	BiasAdd_1T
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dim?
split_1Splitsplit_1/split_dim:output:0BiasAdd_1:output:0*
T0*M
_output_shapes;
9:?????????:?????????:?????????*
	num_split2	
split_1g
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:?????????2
addX
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:?????????2	
Sigmoidk
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:?????????2
add_1^
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:?????????2
	Sigmoid_1d
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:?????????2
mulb
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:?????????2
add_2Q
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:?????????2
Tanh\
mul_1MulSigmoid:y:0init_h*
T0*'
_output_shapes
:?????????2
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
sub/x`
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:?????????2
subZ
mul_2Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:?????????2
mul_2_
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:?????????2
add_3?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileStatelessWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelunstack:output:0recurrent_kernelunstack:output:1*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*S
_output_shapesA
?: : : : :?????????: : :Z:Z:Z:Z* 
_read_only_resource_inputs
 *
bodyR
while_body_15314*
condR
while_cond_15313*R
output_shapesA
?: : : : :?????????: : :Z:Z:Z:Z*
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_2y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  ??2	
runtimel
IdentityIdentitystrided_slice_2:output:0*
T0*'
_output_shapes
:?????????2

Identityt

Identity_1Identitytranspose_1:y:0*
T0*4
_output_shapes"
 :??????????????????2

Identity_1f

Identity_2Identitywhile:output:4*
T0*'
_output_shapes
:?????????2

Identity_2W

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: 2

Identity_3"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*d
_input_shapesS
Q:??????????????????:?????????:Z:Z:Z*<
api_implements*(gru_85420d5a-5eb6-4eb6-bfb4-ef0c67d578e4*
api_preferred_deviceCPU*
go_backwards( *

time_major( :\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinit_h:FB

_output_shapes

:Z
 
_user_specified_namekernel:PL

_output_shapes

:Z
*
_user_specified_namerecurrent_kernel:D@

_output_shapes

:Z

_user_specified_namebias
?&
?
G__inference_sequential_1_layer_call_and_return_conditional_losses_17773

inputs$
 gru_read_readvariableop_resource&
"gru_read_1_readvariableop_resource&
"gru_read_2_readvariableop_resource-
)dense_2_mlcmatmul_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource
identity??dense_2/BiasAdd/ReadVariableOp? dense_2/MLCMatMul/ReadVariableOp?gru/Read/ReadVariableOp?gru/Read_1/ReadVariableOp?gru/Read_2/ReadVariableOpL
	gru/ShapeShapeinputs*
T0*
_output_shapes
:2
	gru/Shape|
gru/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru/strided_slice/stack?
gru/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
gru/strided_slice/stack_1?
gru/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
gru/strided_slice/stack_2?
gru/strided_sliceStridedSlicegru/Shape:output:0 gru/strided_slice/stack:output:0"gru/strided_slice/stack_1:output:0"gru/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
gru/strided_sliced
gru/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
gru/zeros/mul/y|
gru/zeros/mulMulgru/strided_slice:output:0gru/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
gru/zeros/mulg
gru/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
gru/zeros/Less/yw
gru/zeros/LessLessgru/zeros/mul:z:0gru/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
gru/zeros/Lessj
gru/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
gru/zeros/packed/1?
gru/zeros/packedPackgru/strided_slice:output:0gru/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
gru/zeros/packedg
gru/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
gru/zeros/Const?
	gru/zerosFillgru/zeros/packed:output:0gru/zeros/Const:output:0*
T0*'
_output_shapes
:?????????2
	gru/zeros?
gru/Read/ReadVariableOpReadVariableOp gru_read_readvariableop_resource*
_output_shapes

:Z*
dtype02
gru/Read/ReadVariableOpr
gru/IdentityIdentitygru/Read/ReadVariableOp:value:0*
T0*
_output_shapes

:Z2
gru/Identity?
gru/Read_1/ReadVariableOpReadVariableOp"gru_read_1_readvariableop_resource*
_output_shapes

:Z*
dtype02
gru/Read_1/ReadVariableOpx
gru/Identity_1Identity!gru/Read_1/ReadVariableOp:value:0*
T0*
_output_shapes

:Z2
gru/Identity_1?
gru/Read_2/ReadVariableOpReadVariableOp"gru_read_2_readvariableop_resource*
_output_shapes

:Z*
dtype02
gru/Read_2/ReadVariableOpx
gru/Identity_2Identity!gru/Read_2/ReadVariableOp:value:0*
T0*
_output_shapes

:Z2
gru/Identity_2?
gru/PartitionedCallPartitionedCallinputsgru/zeros:output:0gru/Identity:output:0gru/Identity_1:output:0gru/Identity_2:output:0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *T
_output_shapesB
@:?????????:??????????:?????????: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *'
f"R 
__inference_standard_gru_174662
gru/PartitionedCall?
 dense_2/MLCMatMul/ReadVariableOpReadVariableOp)dense_2_mlcmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02"
 dense_2/MLCMatMul/ReadVariableOp?
dense_2/MLCMatMul	MLCMatMulgru/PartitionedCall:output:0(dense_2/MLCMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_2/MLCMatMul?
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
dense_2/BiasAdd/ReadVariableOp?
dense_2/BiasAddBiasAdddense_2/MLCMatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_2/BiasAddq
dense_2/TanhTanhdense_2/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_2/Tanh?
IdentityIdentitydense_2/Tanh:y:0^dense_2/BiasAdd/ReadVariableOp!^dense_2/MLCMatMul/ReadVariableOp^gru/Read/ReadVariableOp^gru/Read_1/ReadVariableOp^gru/Read_2/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????:::::2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2D
 dense_2/MLCMatMul/ReadVariableOp dense_2/MLCMatMul/ReadVariableOp22
gru/Read/ReadVariableOpgru/Read/ReadVariableOp26
gru/Read_1/ReadVariableOpgru/Read_1/ReadVariableOp26
gru/Read_2/ReadVariableOpgru/Read_2/ReadVariableOp:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?2
?
while_body_15314
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_matmul_kernel_0
while_biasadd_unstack_0%
!while_matmul_1_recurrent_kernel_0
while_biasadd_1_unstack_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_matmul_kernel
while_biasadd_unstack#
while_matmul_1_recurrent_kernel
while_biasadd_1_unstack?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
while/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while_matmul_kernel_0*
T0*'
_output_shapes
:?????????Z2
while/MatMul?
while/BiasAddBiasAddwhile/MatMul:product:0while_biasadd_unstack_0*
T0*'
_output_shapes
:?????????Z2
while/BiasAdd\
while/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/Constp
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
while/split/split_dim?
while/splitSplitwhile/split/split_dim:output:0while/BiasAdd:output:0*
T0*M
_output_shapes;
9:?????????:?????????:?????????*
	num_split2
while/split?
while/MatMul_1MatMulwhile_placeholder_2!while_matmul_1_recurrent_kernel_0*
T0*'
_output_shapes
:?????????Z2
while/MatMul_1?
while/BiasAdd_1BiasAddwhile/MatMul_1:product:0while_biasadd_1_unstack_0*
T0*'
_output_shapes
:?????????Z2
while/BiasAdd_1`
while/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
while/Const_1t
while/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
while/split_1/split_dim?
while/split_1Split while/split_1/split_dim:output:0while/BiasAdd_1:output:0*
T0*M
_output_shapes;
9:?????????:?????????:?????????*
	num_split2
while/split_1
	while/addAddV2while/split:output:0while/split_1:output:0*
T0*'
_output_shapes
:?????????2
	while/addj
while/SigmoidSigmoidwhile/add:z:0*
T0*'
_output_shapes
:?????????2
while/Sigmoid?
while/add_1AddV2while/split:output:1while/split_1:output:1*
T0*'
_output_shapes
:?????????2
while/add_1p
while/Sigmoid_1Sigmoidwhile/add_1:z:0*
T0*'
_output_shapes
:?????????2
while/Sigmoid_1|
	while/mulMulwhile/Sigmoid_1:y:0while/split_1:output:2*
T0*'
_output_shapes
:?????????2
	while/mulz
while/add_2AddV2while/split:output:2while/mul:z:0*
T0*'
_output_shapes
:?????????2
while/add_2c

while/TanhTanhwhile/add_2:z:0*
T0*'
_output_shapes
:?????????2

while/Tanh{
while/mul_1Mulwhile/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:?????????2
while/mul_1_
while/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
while/sub/xx
	while/subSubwhile/sub/x:output:0while/Sigmoid:y:0*
T0*'
_output_shapes
:?????????2
	while/subr
while/mul_2Mulwhile/sub:z:0while/Tanh:y:0*
T0*'
_output_shapes
:?????????2
while/mul_2w
while/add_3AddV2while/mul_1:z:0while/mul_2:z:0*
T0*'
_output_shapes
:?????????2
while/add_3?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/add_3:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem`
while/add_4/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_4/yo
while/add_4AddV2while_placeholderwhile/add_4/y:output:0*
T0*
_output_shapes
: 2
while/add_4`
while/add_5/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_5/yv
while/add_5AddV2while_while_loop_counterwhile/add_5/y:output:0*
T0*
_output_shapes
: 2
while/add_5^
while/IdentityIdentitywhile/add_5:z:0*
T0*
_output_shapes
: 2
while/Identityq
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: 2
while/Identity_1b
while/Identity_2Identitywhile/add_4:z:0*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3s
while/Identity_4Identitywhile/add_3:z:0*
T0*'
_output_shapes
:?????????2
while/Identity_4"4
while_biasadd_1_unstackwhile_biasadd_1_unstack_0"0
while_biasadd_unstackwhile_biasadd_unstack_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"D
while_matmul_1_recurrent_kernel!while_matmul_1_recurrent_kernel_0",
while_matmul_kernelwhile_matmul_kernel_0",
while_strided_slicewhile_strided_slice_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*R
_input_shapesA
?: : : : :?????????: : :Z:Z:Z:Z: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:Z: 

_output_shapes
:Z:$	 

_output_shapes

:Z: 


_output_shapes
:Z
?<
?
'__inference_gpu_gru_with_fallback_19956

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3?u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm{
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:??????????2
	transposeb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims/dim}

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2

ExpandDimsP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
splitSplitsplit/split_dim:output:0kernel*
T0*2
_output_shapes 
:::*
	num_split2
splitT
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dim?
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*2
_output_shapes 
:::*
	num_split2	
split_1q
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
Reshape/shapea
ReshapeReshapebiasReshape/shape:output:0*
T0*
_output_shapes	
:?2	
ReshapeT
Const_2Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_2h
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_2/split_dim?
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*8
_output_shapes&
$::::::*
	num_split2	
split_2e
Const_3Const*
_output_shapes
:*
dtype0*
valueB:
?????????2	
Const_3u
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_1/perm{
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes

:2
transpose_1j
	Reshape_1Reshapetranspose_1:y:0Const_3:output:0*
T0*
_output_shapes	
:?2
	Reshape_1u
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_2/perm{
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes

:2
transpose_2j
	Reshape_2Reshapetranspose_2:y:0Const_3:output:0*
T0*
_output_shapes	
:?2
	Reshape_2u
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_3/perm{
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

:2
transpose_3j
	Reshape_3Reshapetranspose_3:y:0Const_3:output:0*
T0*
_output_shapes	
:?2
	Reshape_3u
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_4/perm}
transpose_4	Transposesplit_1:output:0transpose_4/perm:output:0*
T0*
_output_shapes

:2
transpose_4j
	Reshape_4Reshapetranspose_4:y:0Const_3:output:0*
T0*
_output_shapes	
:?2
	Reshape_4u
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_5/perm}
transpose_5	Transposesplit_1:output:1transpose_5/perm:output:0*
T0*
_output_shapes

:2
transpose_5j
	Reshape_5Reshapetranspose_5:y:0Const_3:output:0*
T0*
_output_shapes	
:?2
	Reshape_5u
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_6/perm}
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T0*
_output_shapes

:2
transpose_6j
	Reshape_6Reshapetranspose_6:y:0Const_3:output:0*
T0*
_output_shapes	
:?2
	Reshape_6j
	Reshape_7Reshapesplit_2:output:0Const_3:output:0*
T0*
_output_shapes
:2
	Reshape_7j
	Reshape_8Reshapesplit_2:output:1Const_3:output:0*
T0*
_output_shapes
:2
	Reshape_8j
	Reshape_9Reshapesplit_2:output:2Const_3:output:0*
T0*
_output_shapes
:2
	Reshape_9l

Reshape_10Reshapesplit_2:output:3Const_3:output:0*
T0*
_output_shapes
:2

Reshape_10l

Reshape_11Reshapesplit_2:output:4Const_3:output:0*
T0*
_output_shapes
:2

Reshape_11l

Reshape_12Reshapesplit_2:output:5Const_3:output:0*
T0*
_output_shapes
:2

Reshape_12\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat/axis?
concatConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
N*
T0*
_output_shapes	
:?2
concati
CudnnRNN/input_cConst*
_output_shapes
: *
dtype0*
valueB
 *    2
CudnnRNN/input_c?
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat:output:0*
T0*I
_output_shapes7
5:??????????:?????????: :*
rnn_modegru2

CudnnRNN}
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slicey
transpose_7/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_7/perm?
transpose_7	TransposeCudnnRNN:output:0transpose_7/perm:output:0*
T0*,
_output_shapes
:??????????2
transpose_7{
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:?????????*
squeeze_dims
 2	
Squeezef
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @2	
runtimej
IdentityIdentitystrided_slice:output:0*
T0*'
_output_shapes
:?????????2

Identityl

Identity_1Identitytranspose_7:y:0*
T0*,
_output_shapes
:??????????2

Identity_1h

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:?????????2

Identity_2W

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: 2

Identity_3"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*\
_input_shapesK
I:??????????:?????????:Z:Z:Z*<
api_implements*(gru_c4a462d1-fe50-45b7-baf6-20215ecbec52*
api_preferred_deviceGPU*
go_backwards( *

time_major( :T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinit_h:FB

_output_shapes

:Z
 
_user_specified_namekernel:PL

_output_shapes

:Z
*
_user_specified_namerecurrent_kernel:D@

_output_shapes

:Z

_user_specified_namebias
?2
?
while_body_18826
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_matmul_kernel_0
while_biasadd_unstack_0%
!while_matmul_1_recurrent_kernel_0
while_biasadd_1_unstack_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_matmul_kernel
while_biasadd_unstack#
while_matmul_1_recurrent_kernel
while_biasadd_1_unstack?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
while/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while_matmul_kernel_0*
T0*'
_output_shapes
:?????????Z2
while/MatMul?
while/BiasAddBiasAddwhile/MatMul:product:0while_biasadd_unstack_0*
T0*'
_output_shapes
:?????????Z2
while/BiasAdd\
while/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/Constp
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
while/split/split_dim?
while/splitSplitwhile/split/split_dim:output:0while/BiasAdd:output:0*
T0*M
_output_shapes;
9:?????????:?????????:?????????*
	num_split2
while/split?
while/MatMul_1MatMulwhile_placeholder_2!while_matmul_1_recurrent_kernel_0*
T0*'
_output_shapes
:?????????Z2
while/MatMul_1?
while/BiasAdd_1BiasAddwhile/MatMul_1:product:0while_biasadd_1_unstack_0*
T0*'
_output_shapes
:?????????Z2
while/BiasAdd_1`
while/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
while/Const_1t
while/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
while/split_1/split_dim?
while/split_1Split while/split_1/split_dim:output:0while/BiasAdd_1:output:0*
T0*M
_output_shapes;
9:?????????:?????????:?????????*
	num_split2
while/split_1
	while/addAddV2while/split:output:0while/split_1:output:0*
T0*'
_output_shapes
:?????????2
	while/addj
while/SigmoidSigmoidwhile/add:z:0*
T0*'
_output_shapes
:?????????2
while/Sigmoid?
while/add_1AddV2while/split:output:1while/split_1:output:1*
T0*'
_output_shapes
:?????????2
while/add_1p
while/Sigmoid_1Sigmoidwhile/add_1:z:0*
T0*'
_output_shapes
:?????????2
while/Sigmoid_1|
	while/mulMulwhile/Sigmoid_1:y:0while/split_1:output:2*
T0*'
_output_shapes
:?????????2
	while/mulz
while/add_2AddV2while/split:output:2while/mul:z:0*
T0*'
_output_shapes
:?????????2
while/add_2c

while/TanhTanhwhile/add_2:z:0*
T0*'
_output_shapes
:?????????2

while/Tanh{
while/mul_1Mulwhile/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:?????????2
while/mul_1_
while/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
while/sub/xx
	while/subSubwhile/sub/x:output:0while/Sigmoid:y:0*
T0*'
_output_shapes
:?????????2
	while/subr
while/mul_2Mulwhile/sub:z:0while/Tanh:y:0*
T0*'
_output_shapes
:?????????2
while/mul_2w
while/add_3AddV2while/mul_1:z:0while/mul_2:z:0*
T0*'
_output_shapes
:?????????2
while/add_3?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/add_3:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem`
while/add_4/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_4/yo
while/add_4AddV2while_placeholderwhile/add_4/y:output:0*
T0*
_output_shapes
: 2
while/add_4`
while/add_5/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_5/yv
while/add_5AddV2while_while_loop_counterwhile/add_5/y:output:0*
T0*
_output_shapes
: 2
while/add_5^
while/IdentityIdentitywhile/add_5:z:0*
T0*
_output_shapes
: 2
while/Identityq
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: 2
while/Identity_1b
while/Identity_2Identitywhile/add_4:z:0*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3s
while/Identity_4Identitywhile/add_3:z:0*
T0*'
_output_shapes
:?????????2
while/Identity_4"4
while_biasadd_1_unstackwhile_biasadd_1_unstack_0"0
while_biasadd_unstackwhile_biasadd_unstack_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"D
while_matmul_1_recurrent_kernel!while_matmul_1_recurrent_kernel_0",
while_matmul_kernelwhile_matmul_kernel_0",
while_strided_slicewhile_strided_slice_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*R
_input_shapesA
?: : : : :?????????: : :Z:Z:Z:Z: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:Z: 

_output_shapes
:Z:$	 

_output_shapes

:Z: 


_output_shapes
:Z
?&
?
G__inference_sequential_1_layer_call_and_return_conditional_losses_18249

inputs$
 gru_read_readvariableop_resource&
"gru_read_1_readvariableop_resource&
"gru_read_2_readvariableop_resource-
)dense_2_mlcmatmul_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource
identity??dense_2/BiasAdd/ReadVariableOp? dense_2/MLCMatMul/ReadVariableOp?gru/Read/ReadVariableOp?gru/Read_1/ReadVariableOp?gru/Read_2/ReadVariableOpL
	gru/ShapeShapeinputs*
T0*
_output_shapes
:2
	gru/Shape|
gru/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru/strided_slice/stack?
gru/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
gru/strided_slice/stack_1?
gru/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
gru/strided_slice/stack_2?
gru/strided_sliceStridedSlicegru/Shape:output:0 gru/strided_slice/stack:output:0"gru/strided_slice/stack_1:output:0"gru/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
gru/strided_sliced
gru/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
gru/zeros/mul/y|
gru/zeros/mulMulgru/strided_slice:output:0gru/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
gru/zeros/mulg
gru/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
gru/zeros/Less/yw
gru/zeros/LessLessgru/zeros/mul:z:0gru/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
gru/zeros/Lessj
gru/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
gru/zeros/packed/1?
gru/zeros/packedPackgru/strided_slice:output:0gru/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
gru/zeros/packedg
gru/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
gru/zeros/Const?
	gru/zerosFillgru/zeros/packed:output:0gru/zeros/Const:output:0*
T0*'
_output_shapes
:?????????2
	gru/zeros?
gru/Read/ReadVariableOpReadVariableOp gru_read_readvariableop_resource*
_output_shapes

:Z*
dtype02
gru/Read/ReadVariableOpr
gru/IdentityIdentitygru/Read/ReadVariableOp:value:0*
T0*
_output_shapes

:Z2
gru/Identity?
gru/Read_1/ReadVariableOpReadVariableOp"gru_read_1_readvariableop_resource*
_output_shapes

:Z*
dtype02
gru/Read_1/ReadVariableOpx
gru/Identity_1Identity!gru/Read_1/ReadVariableOp:value:0*
T0*
_output_shapes

:Z2
gru/Identity_1?
gru/Read_2/ReadVariableOpReadVariableOp"gru_read_2_readvariableop_resource*
_output_shapes

:Z*
dtype02
gru/Read_2/ReadVariableOpx
gru/Identity_2Identity!gru/Read_2/ReadVariableOp:value:0*
T0*
_output_shapes

:Z2
gru/Identity_2?
gru/PartitionedCallPartitionedCallinputsgru/zeros:output:0gru/Identity:output:0gru/Identity_1:output:0gru/Identity_2:output:0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *T
_output_shapesB
@:?????????:??????????:?????????: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *'
f"R 
__inference_standard_gru_179422
gru/PartitionedCall?
 dense_2/MLCMatMul/ReadVariableOpReadVariableOp)dense_2_mlcmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02"
 dense_2/MLCMatMul/ReadVariableOp?
dense_2/MLCMatMul	MLCMatMulgru/PartitionedCall:output:0(dense_2/MLCMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_2/MLCMatMul?
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
dense_2/BiasAdd/ReadVariableOp?
dense_2/BiasAddBiasAdddense_2/MLCMatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_2/BiasAddq
dense_2/TanhTanhdense_2/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_2/Tanh?
IdentityIdentitydense_2/Tanh:y:0^dense_2/BiasAdd/ReadVariableOp!^dense_2/MLCMatMul/ReadVariableOp^gru/Read/ReadVariableOp^gru/Read_1/ReadVariableOp^gru/Read_2/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????:::::2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2D
 dense_2/MLCMatMul/ReadVariableOp dense_2/MLCMatMul/ReadVariableOp22
gru/Read/ReadVariableOpgru/Read/ReadVariableOp26
gru/Read_1/ReadVariableOpgru/Read_1/ReadVariableOp26
gru/Read_2/ReadVariableOpgru/Read_2/ReadVariableOp:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
>__inference_gru_layer_call_and_return_conditional_losses_15705

inputs 
read_readvariableop_resource"
read_1_readvariableop_resource"
read_2_readvariableop_resource

identity_3??Read/ReadVariableOp?Read_1/ReadVariableOp?Read_2/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????2
zeros?
Read/ReadVariableOpReadVariableOpread_readvariableop_resource*
_output_shapes

:Z*
dtype02
Read/ReadVariableOpf
IdentityIdentityRead/ReadVariableOp:value:0*
T0*
_output_shapes

:Z2

Identity?
Read_1/ReadVariableOpReadVariableOpread_1_readvariableop_resource*
_output_shapes

:Z*
dtype02
Read_1/ReadVariableOpl

Identity_1IdentityRead_1/ReadVariableOp:value:0*
T0*
_output_shapes

:Z2

Identity_1?
Read_2/ReadVariableOpReadVariableOpread_2_readvariableop_resource*
_output_shapes

:Z*
dtype02
Read_2/ReadVariableOpl

Identity_2IdentityRead_2/ReadVariableOp:value:0*
T0*
_output_shapes

:Z2

Identity_2?
PartitionedCallPartitionedCallinputszeros:output:0Identity:output:0Identity_1:output:0Identity_2:output:0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *\
_output_shapesJ
H:?????????:??????????????????:?????????: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *'
f"R 
__inference_standard_gru_154052
PartitionedCall?

Identity_3IdentityPartitionedCall:output:0^Read/ReadVariableOp^Read_1/ReadVariableOp^Read_2/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_3"!

identity_3Identity_3:output:0*?
_input_shapes.
,:??????????????????:::2*
Read/ReadVariableOpRead/ReadVariableOp2.
Read_1/ReadVariableOpRead_1/ReadVariableOp2.
Read_2/ReadVariableOpRead_2/ReadVariableOp:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
?
>__inference_gru_layer_call_and_return_conditional_losses_16666

inputs 
read_readvariableop_resource"
read_1_readvariableop_resource"
read_2_readvariableop_resource

identity_3??Read/ReadVariableOp?Read_1/ReadVariableOp?Read_2/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????2
zeros?
Read/ReadVariableOpReadVariableOpread_readvariableop_resource*
_output_shapes

:Z*
dtype02
Read/ReadVariableOpf
IdentityIdentityRead/ReadVariableOp:value:0*
T0*
_output_shapes

:Z2

Identity?
Read_1/ReadVariableOpReadVariableOpread_1_readvariableop_resource*
_output_shapes

:Z*
dtype02
Read_1/ReadVariableOpl

Identity_1IdentityRead_1/ReadVariableOp:value:0*
T0*
_output_shapes

:Z2

Identity_1?
Read_2/ReadVariableOpReadVariableOpread_2_readvariableop_resource*
_output_shapes

:Z*
dtype02
Read_2/ReadVariableOpl

Identity_2IdentityRead_2/ReadVariableOp:value:0*
T0*
_output_shapes

:Z2

Identity_2?
PartitionedCallPartitionedCallinputszeros:output:0Identity:output:0Identity_1:output:0Identity_2:output:0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *T
_output_shapesB
@:?????????:??????????:?????????: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *'
f"R 
__inference_standard_gru_163662
PartitionedCall?

Identity_3IdentityPartitionedCall:output:0^Read/ReadVariableOp^Read_1/ReadVariableOp^Read_2/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_3"!

identity_3Identity_3:output:0*7
_input_shapes&
$:??????????:::2*
Read/ReadVariableOpRead/ReadVariableOp2.
Read_1/ReadVariableOpRead_1/ReadVariableOp2.
Read_2/ReadVariableOpRead_2/ReadVariableOp:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
while_cond_15793
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice3
/while_while_cond_15793___redundant_placeholder03
/while_while_cond_15793___redundant_placeholder13
/while_while_cond_15793___redundant_placeholder23
/while_while_cond_15793___redundant_placeholder33
/while_while_cond_15793___redundant_placeholder4
while_identity
n

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*D
_input_shapes3
1: : : : :?????????: :::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
::

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
?
?
>__inference_gru_layer_call_and_return_conditional_losses_19217
inputs_0 
read_readvariableop_resource"
read_1_readvariableop_resource"
read_2_readvariableop_resource

identity_3??Read/ReadVariableOp?Read_1/ReadVariableOp?Read_2/ReadVariableOpF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????2
zeros?
Read/ReadVariableOpReadVariableOpread_readvariableop_resource*
_output_shapes

:Z*
dtype02
Read/ReadVariableOpf
IdentityIdentityRead/ReadVariableOp:value:0*
T0*
_output_shapes

:Z2

Identity?
Read_1/ReadVariableOpReadVariableOpread_1_readvariableop_resource*
_output_shapes

:Z*
dtype02
Read_1/ReadVariableOpl

Identity_1IdentityRead_1/ReadVariableOp:value:0*
T0*
_output_shapes

:Z2

Identity_1?
Read_2/ReadVariableOpReadVariableOpread_2_readvariableop_resource*
_output_shapes

:Z*
dtype02
Read_2/ReadVariableOpl

Identity_2IdentityRead_2/ReadVariableOp:value:0*
T0*
_output_shapes

:Z2

Identity_2?
PartitionedCallPartitionedCallinputs_0zeros:output:0Identity:output:0Identity_1:output:0Identity_2:output:0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *\
_output_shapesJ
H:?????????:??????????????????:?????????: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *'
f"R 
__inference_standard_gru_189172
PartitionedCall?

Identity_3IdentityPartitionedCall:output:0^Read/ReadVariableOp^Read_1/ReadVariableOp^Read_2/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_3"!

identity_3Identity_3:output:0*?
_input_shapes.
,:??????????????????:::2*
Read/ReadVariableOpRead/ReadVariableOp2.
Read_1/ReadVariableOpRead_1/ReadVariableOp2.
Read_2/ReadVariableOpRead_2/ReadVariableOp:^ Z
4
_output_shapes"
 :??????????????????
"
_user_specified_name
inputs/0
?	
?
while_cond_13895
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice3
/while_while_cond_13895___redundant_placeholder03
/while_while_cond_13895___redundant_placeholder13
/while_while_cond_13895___redundant_placeholder23
/while_while_cond_13895___redundant_placeholder33
/while_while_cond_13895___redundant_placeholder4
while_identity
n

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*D
_input_shapes3
1: : : : :?????????: :::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
::

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
?E
?
__inference_standard_gru_13987

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3X
unstackUnpackbias*
T0* 
_output_shapes
:Z:Z*	
num2	
unstacku
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm{
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:??????????2
	transposeK
ShapeShapetranspose:y:0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_1n
MatMulMatMulstrided_slice_1:output:0kernel*
T0*'
_output_shapes
:?????????Z2
MatMuls
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*'
_output_shapes
:?????????Z2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:?????????:?????????:?????????*
	num_split2
splitj
MatMul_1MatMulinit_hrecurrent_kernel*
T0*'
_output_shapes
:?????????Z2

MatMul_1y
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*'
_output_shapes
:?????????Z2
	BiasAdd_1T
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dim?
split_1Splitsplit_1/split_dim:output:0BiasAdd_1:output:0*
T0*M
_output_shapes;
9:?????????:?????????:?????????*
	num_split2	
split_1g
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:?????????2
addX
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:?????????2	
Sigmoidk
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:?????????2
add_1^
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:?????????2
	Sigmoid_1d
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:?????????2
mulb
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:?????????2
add_2Q
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:?????????2
Tanh\
mul_1MulSigmoid:y:0init_h*
T0*'
_output_shapes
:?????????2
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
sub/x`
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:?????????2
subZ
mul_2Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:?????????2
mul_2_
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:?????????2
add_3?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileStatelessWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelunstack:output:0recurrent_kernelunstack:output:1*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*S
_output_shapesA
?: : : : :?????????: : :Z:Z:Z:Z* 
_read_only_resource_inputs
 *
bodyR
while_body_13896*
condR
while_cond_13895*R
output_shapesA
?: : : : :?????????: : :Z:Z:Z:Z*
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_2y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:??????????2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  ??2	
runtimel
IdentityIdentitystrided_slice_2:output:0*
T0*'
_output_shapes
:?????????2

Identityl

Identity_1Identitytranspose_1:y:0*
T0*,
_output_shapes
:??????????2

Identity_1f

Identity_2Identitywhile:output:4*
T0*'
_output_shapes
:?????????2

Identity_2W

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: 2

Identity_3"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*\
_input_shapesK
I:??????????:?????????:Z:Z:Z*<
api_implements*(gru_b74c1e92-5c0b-403e-bc2f-dc8926149f3e*
api_preferred_deviceCPU*
go_backwards( *

time_major( :T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinit_h:FB

_output_shapes

:Z
 
_user_specified_namekernel:PL

_output_shapes

:Z
*
_user_specified_namerecurrent_kernel:D@

_output_shapes

:Z

_user_specified_namebias
?2
?
while_body_16744
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_matmul_kernel_0
while_biasadd_unstack_0%
!while_matmul_1_recurrent_kernel_0
while_biasadd_1_unstack_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_matmul_kernel
while_biasadd_unstack#
while_matmul_1_recurrent_kernel
while_biasadd_1_unstack?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
while/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while_matmul_kernel_0*
T0*'
_output_shapes
:?????????Z2
while/MatMul?
while/BiasAddBiasAddwhile/MatMul:product:0while_biasadd_unstack_0*
T0*'
_output_shapes
:?????????Z2
while/BiasAdd\
while/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/Constp
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
while/split/split_dim?
while/splitSplitwhile/split/split_dim:output:0while/BiasAdd:output:0*
T0*M
_output_shapes;
9:?????????:?????????:?????????*
	num_split2
while/split?
while/MatMul_1MatMulwhile_placeholder_2!while_matmul_1_recurrent_kernel_0*
T0*'
_output_shapes
:?????????Z2
while/MatMul_1?
while/BiasAdd_1BiasAddwhile/MatMul_1:product:0while_biasadd_1_unstack_0*
T0*'
_output_shapes
:?????????Z2
while/BiasAdd_1`
while/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
while/Const_1t
while/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
while/split_1/split_dim?
while/split_1Split while/split_1/split_dim:output:0while/BiasAdd_1:output:0*
T0*M
_output_shapes;
9:?????????:?????????:?????????*
	num_split2
while/split_1
	while/addAddV2while/split:output:0while/split_1:output:0*
T0*'
_output_shapes
:?????????2
	while/addj
while/SigmoidSigmoidwhile/add:z:0*
T0*'
_output_shapes
:?????????2
while/Sigmoid?
while/add_1AddV2while/split:output:1while/split_1:output:1*
T0*'
_output_shapes
:?????????2
while/add_1p
while/Sigmoid_1Sigmoidwhile/add_1:z:0*
T0*'
_output_shapes
:?????????2
while/Sigmoid_1|
	while/mulMulwhile/Sigmoid_1:y:0while/split_1:output:2*
T0*'
_output_shapes
:?????????2
	while/mulz
while/add_2AddV2while/split:output:2while/mul:z:0*
T0*'
_output_shapes
:?????????2
while/add_2c

while/TanhTanhwhile/add_2:z:0*
T0*'
_output_shapes
:?????????2

while/Tanh{
while/mul_1Mulwhile/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:?????????2
while/mul_1_
while/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
while/sub/xx
	while/subSubwhile/sub/x:output:0while/Sigmoid:y:0*
T0*'
_output_shapes
:?????????2
	while/subr
while/mul_2Mulwhile/sub:z:0while/Tanh:y:0*
T0*'
_output_shapes
:?????????2
while/mul_2w
while/add_3AddV2while/mul_1:z:0while/mul_2:z:0*
T0*'
_output_shapes
:?????????2
while/add_3?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/add_3:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem`
while/add_4/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_4/yo
while/add_4AddV2while_placeholderwhile/add_4/y:output:0*
T0*
_output_shapes
: 2
while/add_4`
while/add_5/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_5/yv
while/add_5AddV2while_while_loop_counterwhile/add_5/y:output:0*
T0*
_output_shapes
: 2
while/add_5^
while/IdentityIdentitywhile/add_5:z:0*
T0*
_output_shapes
: 2
while/Identityq
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: 2
while/Identity_1b
while/Identity_2Identitywhile/add_4:z:0*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3s
while/Identity_4Identitywhile/add_3:z:0*
T0*'
_output_shapes
:?????????2
while/Identity_4"4
while_biasadd_1_unstackwhile_biasadd_1_unstack_0"0
while_biasadd_unstackwhile_biasadd_unstack_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"D
while_matmul_1_recurrent_kernel!while_matmul_1_recurrent_kernel_0",
while_matmul_kernelwhile_matmul_kernel_0",
while_strided_slicewhile_strided_slice_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*R
_input_shapesA
?: : : : :?????????: : :Z:Z:Z:Z: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:Z: 

_output_shapes
:Z:$	 

_output_shapes

:Z: 


_output_shapes
:Z
?
?
>__inference_gru_layer_call_and_return_conditional_losses_20177

inputs 
read_readvariableop_resource"
read_1_readvariableop_resource"
read_2_readvariableop_resource

identity_3??Read/ReadVariableOp?Read_1/ReadVariableOp?Read_2/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????2
zeros?
Read/ReadVariableOpReadVariableOpread_readvariableop_resource*
_output_shapes

:Z*
dtype02
Read/ReadVariableOpf
IdentityIdentityRead/ReadVariableOp:value:0*
T0*
_output_shapes

:Z2

Identity?
Read_1/ReadVariableOpReadVariableOpread_1_readvariableop_resource*
_output_shapes

:Z*
dtype02
Read_1/ReadVariableOpl

Identity_1IdentityRead_1/ReadVariableOp:value:0*
T0*
_output_shapes

:Z2

Identity_1?
Read_2/ReadVariableOpReadVariableOpread_2_readvariableop_resource*
_output_shapes

:Z*
dtype02
Read_2/ReadVariableOpl

Identity_2IdentityRead_2/ReadVariableOp:value:0*
T0*
_output_shapes

:Z2

Identity_2?
PartitionedCallPartitionedCallinputszeros:output:0Identity:output:0Identity_1:output:0Identity_2:output:0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *T
_output_shapesB
@:?????????:??????????:?????????: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *'
f"R 
__inference_standard_gru_198772
PartitionedCall?

Identity_3IdentityPartitionedCall:output:0^Read/ReadVariableOp^Read_1/ReadVariableOp^Read_2/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_3"!

identity_3Identity_3:output:0*7
_input_shapes&
$:??????????:::2*
Read/ReadVariableOpRead/ReadVariableOp2.
Read_1/ReadVariableOpRead_1/ReadVariableOp2.
Read_2/ReadVariableOpRead_2/ReadVariableOp:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
,__inference_sequential_1_layer_call_fn_18264

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_172282
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????:::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?=
?
'__inference_gpu_gru_with_fallback_18527

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3?u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????2
	transposeb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims/dim}

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2

ExpandDimsP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
splitSplitsplit/split_dim:output:0kernel*
T0*2
_output_shapes 
:::*
	num_split2
splitT
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dim?
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*2
_output_shapes 
:::*
	num_split2	
split_1q
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
Reshape/shapea
ReshapeReshapebiasReshape/shape:output:0*
T0*
_output_shapes	
:?2	
ReshapeT
Const_2Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_2h
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_2/split_dim?
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*8
_output_shapes&
$::::::*
	num_split2	
split_2e
Const_3Const*
_output_shapes
:*
dtype0*
valueB:
?????????2	
Const_3u
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_1/perm{
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes

:2
transpose_1j
	Reshape_1Reshapetranspose_1:y:0Const_3:output:0*
T0*
_output_shapes	
:?2
	Reshape_1u
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_2/perm{
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes

:2
transpose_2j
	Reshape_2Reshapetranspose_2:y:0Const_3:output:0*
T0*
_output_shapes	
:?2
	Reshape_2u
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_3/perm{
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

:2
transpose_3j
	Reshape_3Reshapetranspose_3:y:0Const_3:output:0*
T0*
_output_shapes	
:?2
	Reshape_3u
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_4/perm}
transpose_4	Transposesplit_1:output:0transpose_4/perm:output:0*
T0*
_output_shapes

:2
transpose_4j
	Reshape_4Reshapetranspose_4:y:0Const_3:output:0*
T0*
_output_shapes	
:?2
	Reshape_4u
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_5/perm}
transpose_5	Transposesplit_1:output:1transpose_5/perm:output:0*
T0*
_output_shapes

:2
transpose_5j
	Reshape_5Reshapetranspose_5:y:0Const_3:output:0*
T0*
_output_shapes	
:?2
	Reshape_5u
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_6/perm}
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T0*
_output_shapes

:2
transpose_6j
	Reshape_6Reshapetranspose_6:y:0Const_3:output:0*
T0*
_output_shapes	
:?2
	Reshape_6j
	Reshape_7Reshapesplit_2:output:0Const_3:output:0*
T0*
_output_shapes
:2
	Reshape_7j
	Reshape_8Reshapesplit_2:output:1Const_3:output:0*
T0*
_output_shapes
:2
	Reshape_8j
	Reshape_9Reshapesplit_2:output:2Const_3:output:0*
T0*
_output_shapes
:2
	Reshape_9l

Reshape_10Reshapesplit_2:output:3Const_3:output:0*
T0*
_output_shapes
:2

Reshape_10l

Reshape_11Reshapesplit_2:output:4Const_3:output:0*
T0*
_output_shapes
:2

Reshape_11l

Reshape_12Reshapesplit_2:output:5Const_3:output:0*
T0*
_output_shapes
:2

Reshape_12\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat/axis?
concatConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
N*
T0*
_output_shapes	
:?2
concati
CudnnRNN/input_cConst*
_output_shapes
: *
dtype0*
valueB
 *    2
CudnnRNN/input_c?
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat:output:0*
T0*Q
_output_shapes?
=:??????????????????:?????????: :*
rnn_modegru2

CudnnRNN}
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slicey
transpose_7/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_7/perm?
transpose_7	TransposeCudnnRNN:output:0transpose_7/perm:output:0*
T0*4
_output_shapes"
 :??????????????????2
transpose_7{
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:?????????*
squeeze_dims
 2	
Squeezef
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @2	
runtimej
IdentityIdentitystrided_slice:output:0*
T0*'
_output_shapes
:?????????2

Identityt

Identity_1Identitytranspose_7:y:0*
T0*4
_output_shapes"
 :??????????????????2

Identity_1h

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:?????????2

Identity_2W

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: 2

Identity_3"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*d
_input_shapesS
Q:??????????????????:?????????:Z:Z:Z*<
api_implements*(gru_7fd55799-5a12-43e2-bc93-7077640e6183*
api_preferred_deviceGPU*
go_backwards( *

time_major( :\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinit_h:FB

_output_shapes

:Z
 
_user_specified_namekernel:PL

_output_shapes

:Z
*
_user_specified_namerecurrent_kernel:D@

_output_shapes

:Z

_user_specified_namebias
?E
?
__inference_standard_gru_17466

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3X
unstackUnpackbias*
T0* 
_output_shapes
:Z:Z*	
num2	
unstacku
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm{
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:??????????2
	transposeK
ShapeShapetranspose:y:0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_1n
MatMulMatMulstrided_slice_1:output:0kernel*
T0*'
_output_shapes
:?????????Z2
MatMuls
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*'
_output_shapes
:?????????Z2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:?????????:?????????:?????????*
	num_split2
splitj
MatMul_1MatMulinit_hrecurrent_kernel*
T0*'
_output_shapes
:?????????Z2

MatMul_1y
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*'
_output_shapes
:?????????Z2
	BiasAdd_1T
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dim?
split_1Splitsplit_1/split_dim:output:0BiasAdd_1:output:0*
T0*M
_output_shapes;
9:?????????:?????????:?????????*
	num_split2	
split_1g
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:?????????2
addX
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:?????????2	
Sigmoidk
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:?????????2
add_1^
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:?????????2
	Sigmoid_1d
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:?????????2
mulb
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:?????????2
add_2Q
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:?????????2
Tanh\
mul_1MulSigmoid:y:0init_h*
T0*'
_output_shapes
:?????????2
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
sub/x`
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:?????????2
subZ
mul_2Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:?????????2
mul_2_
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:?????????2
add_3?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileStatelessWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelunstack:output:0recurrent_kernelunstack:output:1*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*S
_output_shapesA
?: : : : :?????????: : :Z:Z:Z:Z* 
_read_only_resource_inputs
 *
bodyR
while_body_17375*
condR
while_cond_17374*R
output_shapesA
?: : : : :?????????: : :Z:Z:Z:Z*
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_2y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:??????????2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  ??2	
runtimel
IdentityIdentitystrided_slice_2:output:0*
T0*'
_output_shapes
:?????????2

Identityl

Identity_1Identitytranspose_1:y:0*
T0*,
_output_shapes
:??????????2

Identity_1f

Identity_2Identitywhile:output:4*
T0*'
_output_shapes
:?????????2

Identity_2W

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: 2

Identity_3"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*\
_input_shapesK
I:??????????:?????????:Z:Z:Z*<
api_implements*(gru_5f9dfa88-82d0-4c61-bd4c-33d58af38bad*
api_preferred_deviceCPU*
go_backwards( *

time_major( :T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinit_h:FB

_output_shapes

:Z
 
_user_specified_namekernel:PL

_output_shapes

:Z
*
_user_specified_namerecurrent_kernel:D@

_output_shapes

:Z

_user_specified_namebias
?
?
>__inference_gru_layer_call_and_return_conditional_losses_16185

inputs 
read_readvariableop_resource"
read_1_readvariableop_resource"
read_2_readvariableop_resource

identity_3??Read/ReadVariableOp?Read_1/ReadVariableOp?Read_2/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????2
zeros?
Read/ReadVariableOpReadVariableOpread_readvariableop_resource*
_output_shapes

:Z*
dtype02
Read/ReadVariableOpf
IdentityIdentityRead/ReadVariableOp:value:0*
T0*
_output_shapes

:Z2

Identity?
Read_1/ReadVariableOpReadVariableOpread_1_readvariableop_resource*
_output_shapes

:Z*
dtype02
Read_1/ReadVariableOpl

Identity_1IdentityRead_1/ReadVariableOp:value:0*
T0*
_output_shapes

:Z2

Identity_1?
Read_2/ReadVariableOpReadVariableOpread_2_readvariableop_resource*
_output_shapes

:Z*
dtype02
Read_2/ReadVariableOpl

Identity_2IdentityRead_2/ReadVariableOp:value:0*
T0*
_output_shapes

:Z2

Identity_2?
PartitionedCallPartitionedCallinputszeros:output:0Identity:output:0Identity_1:output:0Identity_2:output:0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *\
_output_shapesJ
H:?????????:??????????????????:?????????: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *'
f"R 
__inference_standard_gru_158852
PartitionedCall?

Identity_3IdentityPartitionedCall:output:0^Read/ReadVariableOp^Read_1/ReadVariableOp^Read_2/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_3"!

identity_3Identity_3:output:0*?
_input_shapes.
,:??????????????????:::2*
Read/ReadVariableOpRead/ReadVariableOp2.
Read_1/ReadVariableOpRead_1/ReadVariableOp2.
Read_2/ReadVariableOpRead_2/ReadVariableOp:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?M
?
%__forward_gpu_gru_with_fallback_20174

inputs
init_h_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3
cudnnrnn
strided_slice
transpose_7_perm
transpose_7

cudnnrnn_0

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
cudnnrnn_input_c

concat
transpose_perm

init_h
concat_axis
	reshape_1
	reshape_2
	reshape_3
	reshape_4
	reshape_5
	reshape_6
	reshape_7
	reshape_8
	reshape_9

reshape_10

reshape_11

reshape_12
transpose_1_perm
transpose_1
transpose_2_perm
transpose_2
transpose_3_perm
transpose_3
transpose_4_perm
transpose_4
transpose_5_perm
transpose_5
transpose_6_perm
transpose_6
split_2_split_dim
split_split_dim	
split
split_1_split_dim
split_1
reshape?u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permO
transpose_0	Transposeinputstranspose/perm:output:0*
T02
	transposeb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims/dim

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2

ExpandDimsP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dimU
split_0Splitsplit/split_dim:output:0kernel*
T0*
	num_split2
splitT
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dime
	split_1_0Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*
	num_split2	
split_1q
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
Reshape/shapea
ReshapeReshapebiasReshape/shape:output:0*
T0*
_output_shapes	
:?2	
ReshapeT
Const_2Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_2h
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_2/split_dim?
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*8
_output_shapes&
$::::::*
	num_split2	
split_2e
Const_3Const*
_output_shapes
:*
dtype0*
valueB:
?????????2	
Const_3u
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_1/perm_
transpose_1_0	Transposesplit_0:output:0transpose_1/perm:output:0*
T02
transpose_1l
	Reshape_1Reshapetranspose_1_0:y:0Const_3:output:0*
T0*
_output_shapes	
:?2
	Reshape_1u
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_2/perm_
transpose_2_0	Transposesplit_0:output:1transpose_2/perm:output:0*
T02
transpose_2l
	Reshape_2Reshapetranspose_2_0:y:0Const_3:output:0*
T0*
_output_shapes	
:?2
	Reshape_2u
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_3/perm_
transpose_3_0	Transposesplit_0:output:2transpose_3/perm:output:0*
T02
transpose_3l
	Reshape_3Reshapetranspose_3_0:y:0Const_3:output:0*
T0*
_output_shapes	
:?2
	Reshape_3u
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_4/perma
transpose_4_0	Transposesplit_1_0:output:0transpose_4/perm:output:0*
T02
transpose_4l
	Reshape_4Reshapetranspose_4_0:y:0Const_3:output:0*
T0*
_output_shapes	
:?2
	Reshape_4u
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_5/perma
transpose_5_0	Transposesplit_1_0:output:1transpose_5/perm:output:0*
T02
transpose_5l
	Reshape_5Reshapetranspose_5_0:y:0Const_3:output:0*
T0*
_output_shapes	
:?2
	Reshape_5u
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_6/perma
transpose_6_0	Transposesplit_1_0:output:2transpose_6/perm:output:0*
T02
transpose_6l
	Reshape_6Reshapetranspose_6_0:y:0Const_3:output:0*
T0*
_output_shapes	
:?2
	Reshape_6j
	Reshape_7Reshapesplit_2:output:0Const_3:output:0*
T0*
_output_shapes
:2
	Reshape_7j
	Reshape_8Reshapesplit_2:output:1Const_3:output:0*
T0*
_output_shapes
:2
	Reshape_8j
	Reshape_9Reshapesplit_2:output:2Const_3:output:0*
T0*
_output_shapes
:2
	Reshape_9l

Reshape_10Reshapesplit_2:output:3Const_3:output:0*
T0*
_output_shapes
:2

Reshape_10l

Reshape_11Reshapesplit_2:output:4Const_3:output:0*
T0*
_output_shapes
:2

Reshape_11l

Reshape_12Reshapesplit_2:output:5Const_3:output:0*
T0*
_output_shapes
:2

Reshape_12\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat/axis?
concat_0ConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
N*
T02
concati
CudnnRNN/input_cConst*
_output_shapes
: *
dtype0*
valueB
 *    2
CudnnRNN/input_c?
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat_0:output:0*
T0*I
_output_shapes7
5:??????????:?????????: :*
rnn_modegru2

CudnnRNN}
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_slice_0StridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
shrink_axis_mask2
strided_slicey
transpose_7/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_7/perm`
transpose_7_0	TransposeCudnnRNN:output:0transpose_7/perm:output:0*
T02
transpose_7{
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:?????????*
squeeze_dims
 2	
Squeezef
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @2	
runtimel
IdentityIdentitystrided_slice_0:output:0*
T0*'
_output_shapes
:?????????2

Identityn

Identity_1Identitytranspose_7_0:y:0*
T0*,
_output_shapes
:??????????2

Identity_1h

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:?????????2

Identity_2W

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: 2

Identity_3"
concatconcat_0:output:0"#
concat_axisconcat/axis:output:0"
cudnnrnnCudnnRNN:output:0"!

cudnnrnn_0CudnnRNN:output_h:0"!

cudnnrnn_1CudnnRNN:output_c:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"-
cudnnrnn_input_cCudnnRNN/input_c:output:0"!

expanddimsExpandDims:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"
init_hinit_h_0"
reshapeReshape:output:0"
	reshape_1Reshape_1:output:0"!

reshape_10Reshape_10:output:0"!

reshape_11Reshape_11:output:0"!

reshape_12Reshape_12:output:0"
	reshape_2Reshape_2:output:0"
	reshape_3Reshape_3:output:0"
	reshape_4Reshape_4:output:0"
	reshape_5Reshape_5:output:0"
	reshape_6Reshape_6:output:0"
	reshape_7Reshape_7:output:0"
	reshape_8Reshape_8:output:0"
	reshape_9Reshape_9:output:0"
splitsplit_0:output:0"
split_1split_1_0:output:0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0")
strided_slicestrided_slice_0:output:0"
	transposetranspose_0:y:0" 
transpose_1transpose_1_0:y:0"-
transpose_1_permtranspose_1/perm:output:0" 
transpose_2transpose_2_0:y:0"-
transpose_2_permtranspose_2/perm:output:0" 
transpose_3transpose_3_0:y:0"-
transpose_3_permtranspose_3/perm:output:0" 
transpose_4transpose_4_0:y:0"-
transpose_4_permtranspose_4/perm:output:0" 
transpose_5transpose_5_0:y:0"-
transpose_5_permtranspose_5/perm:output:0" 
transpose_6transpose_6_0:y:0"-
transpose_6_permtranspose_6/perm:output:0" 
transpose_7transpose_7_0:y:0"-
transpose_7_permtranspose_7/perm:output:0")
transpose_permtranspose/perm:output:0*\
_input_shapesK
I:??????????:?????????:Z:Z:Z*<
api_implements*(gru_c4a462d1-fe50-45b7-baf6-20215ecbec52*
api_preferred_deviceGPU*T
backward_function_name:8__inference___backward_gpu_gru_with_fallback_19957_20175*
go_backwards( *

time_major( :T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinit_h:FB

_output_shapes

:Z
 
_user_specified_namekernel:PL

_output_shapes

:Z
*
_user_specified_namerecurrent_kernel:D@

_output_shapes

:Z

_user_specified_namebias
?=
?
'__inference_gpu_gru_with_fallback_18996

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3?u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????2
	transposeb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims/dim}

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2

ExpandDimsP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
splitSplitsplit/split_dim:output:0kernel*
T0*2
_output_shapes 
:::*
	num_split2
splitT
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dim?
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*2
_output_shapes 
:::*
	num_split2	
split_1q
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
Reshape/shapea
ReshapeReshapebiasReshape/shape:output:0*
T0*
_output_shapes	
:?2	
ReshapeT
Const_2Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_2h
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_2/split_dim?
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*8
_output_shapes&
$::::::*
	num_split2	
split_2e
Const_3Const*
_output_shapes
:*
dtype0*
valueB:
?????????2	
Const_3u
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_1/perm{
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes

:2
transpose_1j
	Reshape_1Reshapetranspose_1:y:0Const_3:output:0*
T0*
_output_shapes	
:?2
	Reshape_1u
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_2/perm{
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes

:2
transpose_2j
	Reshape_2Reshapetranspose_2:y:0Const_3:output:0*
T0*
_output_shapes	
:?2
	Reshape_2u
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_3/perm{
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

:2
transpose_3j
	Reshape_3Reshapetranspose_3:y:0Const_3:output:0*
T0*
_output_shapes	
:?2
	Reshape_3u
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_4/perm}
transpose_4	Transposesplit_1:output:0transpose_4/perm:output:0*
T0*
_output_shapes

:2
transpose_4j
	Reshape_4Reshapetranspose_4:y:0Const_3:output:0*
T0*
_output_shapes	
:?2
	Reshape_4u
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_5/perm}
transpose_5	Transposesplit_1:output:1transpose_5/perm:output:0*
T0*
_output_shapes

:2
transpose_5j
	Reshape_5Reshapetranspose_5:y:0Const_3:output:0*
T0*
_output_shapes	
:?2
	Reshape_5u
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_6/perm}
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T0*
_output_shapes

:2
transpose_6j
	Reshape_6Reshapetranspose_6:y:0Const_3:output:0*
T0*
_output_shapes	
:?2
	Reshape_6j
	Reshape_7Reshapesplit_2:output:0Const_3:output:0*
T0*
_output_shapes
:2
	Reshape_7j
	Reshape_8Reshapesplit_2:output:1Const_3:output:0*
T0*
_output_shapes
:2
	Reshape_8j
	Reshape_9Reshapesplit_2:output:2Const_3:output:0*
T0*
_output_shapes
:2
	Reshape_9l

Reshape_10Reshapesplit_2:output:3Const_3:output:0*
T0*
_output_shapes
:2

Reshape_10l

Reshape_11Reshapesplit_2:output:4Const_3:output:0*
T0*
_output_shapes
:2

Reshape_11l

Reshape_12Reshapesplit_2:output:5Const_3:output:0*
T0*
_output_shapes
:2

Reshape_12\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat/axis?
concatConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
N*
T0*
_output_shapes	
:?2
concati
CudnnRNN/input_cConst*
_output_shapes
: *
dtype0*
valueB
 *    2
CudnnRNN/input_c?
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat:output:0*
T0*Q
_output_shapes?
=:??????????????????:?????????: :*
rnn_modegru2

CudnnRNN}
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slicey
transpose_7/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_7/perm?
transpose_7	TransposeCudnnRNN:output:0transpose_7/perm:output:0*
T0*4
_output_shapes"
 :??????????????????2
transpose_7{
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:?????????*
squeeze_dims
 2	
Squeezef
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @2	
runtimej
IdentityIdentitystrided_slice:output:0*
T0*'
_output_shapes
:?????????2

Identityt

Identity_1Identitytranspose_7:y:0*
T0*4
_output_shapes"
 :??????????????????2

Identity_1h

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:?????????2

Identity_2W

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: 2

Identity_3"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*d
_input_shapesS
Q:??????????????????:?????????:Z:Z:Z*<
api_implements*(gru_d29f5daa-115b-49e1-809f-181a86e00a5d*
api_preferred_deviceGPU*
go_backwards( *

time_major( :\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinit_h:FB

_output_shapes

:Z
 
_user_specified_namekernel:PL

_output_shapes

:Z
*
_user_specified_namerecurrent_kernel:D@

_output_shapes

:Z

_user_specified_namebias
?M
?
%__forward_gpu_gru_with_fallback_17132

inputs
init_h_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3
cudnnrnn
strided_slice
transpose_7_perm
transpose_7

cudnnrnn_0

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
cudnnrnn_input_c

concat
transpose_perm

init_h
concat_axis
	reshape_1
	reshape_2
	reshape_3
	reshape_4
	reshape_5
	reshape_6
	reshape_7
	reshape_8
	reshape_9

reshape_10

reshape_11

reshape_12
transpose_1_perm
transpose_1
transpose_2_perm
transpose_2
transpose_3_perm
transpose_3
transpose_4_perm
transpose_4
transpose_5_perm
transpose_5
transpose_6_perm
transpose_6
split_2_split_dim
split_split_dim	
split
split_1_split_dim
split_1
reshape?u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permO
transpose_0	Transposeinputstranspose/perm:output:0*
T02
	transposeb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims/dim

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2

ExpandDimsP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dimU
split_0Splitsplit/split_dim:output:0kernel*
T0*
	num_split2
splitT
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dime
	split_1_0Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*
	num_split2	
split_1q
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
Reshape/shapea
ReshapeReshapebiasReshape/shape:output:0*
T0*
_output_shapes	
:?2	
ReshapeT
Const_2Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_2h
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_2/split_dim?
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*8
_output_shapes&
$::::::*
	num_split2	
split_2e
Const_3Const*
_output_shapes
:*
dtype0*
valueB:
?????????2	
Const_3u
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_1/perm_
transpose_1_0	Transposesplit_0:output:0transpose_1/perm:output:0*
T02
transpose_1l
	Reshape_1Reshapetranspose_1_0:y:0Const_3:output:0*
T0*
_output_shapes	
:?2
	Reshape_1u
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_2/perm_
transpose_2_0	Transposesplit_0:output:1transpose_2/perm:output:0*
T02
transpose_2l
	Reshape_2Reshapetranspose_2_0:y:0Const_3:output:0*
T0*
_output_shapes	
:?2
	Reshape_2u
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_3/perm_
transpose_3_0	Transposesplit_0:output:2transpose_3/perm:output:0*
T02
transpose_3l
	Reshape_3Reshapetranspose_3_0:y:0Const_3:output:0*
T0*
_output_shapes	
:?2
	Reshape_3u
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_4/perma
transpose_4_0	Transposesplit_1_0:output:0transpose_4/perm:output:0*
T02
transpose_4l
	Reshape_4Reshapetranspose_4_0:y:0Const_3:output:0*
T0*
_output_shapes	
:?2
	Reshape_4u
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_5/perma
transpose_5_0	Transposesplit_1_0:output:1transpose_5/perm:output:0*
T02
transpose_5l
	Reshape_5Reshapetranspose_5_0:y:0Const_3:output:0*
T0*
_output_shapes	
:?2
	Reshape_5u
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_6/perma
transpose_6_0	Transposesplit_1_0:output:2transpose_6/perm:output:0*
T02
transpose_6l
	Reshape_6Reshapetranspose_6_0:y:0Const_3:output:0*
T0*
_output_shapes	
:?2
	Reshape_6j
	Reshape_7Reshapesplit_2:output:0Const_3:output:0*
T0*
_output_shapes
:2
	Reshape_7j
	Reshape_8Reshapesplit_2:output:1Const_3:output:0*
T0*
_output_shapes
:2
	Reshape_8j
	Reshape_9Reshapesplit_2:output:2Const_3:output:0*
T0*
_output_shapes
:2
	Reshape_9l

Reshape_10Reshapesplit_2:output:3Const_3:output:0*
T0*
_output_shapes
:2

Reshape_10l

Reshape_11Reshapesplit_2:output:4Const_3:output:0*
T0*
_output_shapes
:2

Reshape_11l

Reshape_12Reshapesplit_2:output:5Const_3:output:0*
T0*
_output_shapes
:2

Reshape_12\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat/axis?
concat_0ConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
N*
T02
concati
CudnnRNN/input_cConst*
_output_shapes
: *
dtype0*
valueB
 *    2
CudnnRNN/input_c?
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat_0:output:0*
T0*I
_output_shapes7
5:??????????:?????????: :*
rnn_modegru2

CudnnRNN}
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_slice_0StridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
shrink_axis_mask2
strided_slicey
transpose_7/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_7/perm`
transpose_7_0	TransposeCudnnRNN:output:0transpose_7/perm:output:0*
T02
transpose_7{
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:?????????*
squeeze_dims
 2	
Squeezef
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @2	
runtimel
IdentityIdentitystrided_slice_0:output:0*
T0*'
_output_shapes
:?????????2

Identityn

Identity_1Identitytranspose_7_0:y:0*
T0*,
_output_shapes
:??????????2

Identity_1h

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:?????????2

Identity_2W

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: 2

Identity_3"
concatconcat_0:output:0"#
concat_axisconcat/axis:output:0"
cudnnrnnCudnnRNN:output:0"!

cudnnrnn_0CudnnRNN:output_h:0"!

cudnnrnn_1CudnnRNN:output_c:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"-
cudnnrnn_input_cCudnnRNN/input_c:output:0"!

expanddimsExpandDims:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"
init_hinit_h_0"
reshapeReshape:output:0"
	reshape_1Reshape_1:output:0"!

reshape_10Reshape_10:output:0"!

reshape_11Reshape_11:output:0"!

reshape_12Reshape_12:output:0"
	reshape_2Reshape_2:output:0"
	reshape_3Reshape_3:output:0"
	reshape_4Reshape_4:output:0"
	reshape_5Reshape_5:output:0"
	reshape_6Reshape_6:output:0"
	reshape_7Reshape_7:output:0"
	reshape_8Reshape_8:output:0"
	reshape_9Reshape_9:output:0"
splitsplit_0:output:0"
split_1split_1_0:output:0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0")
strided_slicestrided_slice_0:output:0"
	transposetranspose_0:y:0" 
transpose_1transpose_1_0:y:0"-
transpose_1_permtranspose_1/perm:output:0" 
transpose_2transpose_2_0:y:0"-
transpose_2_permtranspose_2/perm:output:0" 
transpose_3transpose_3_0:y:0"-
transpose_3_permtranspose_3/perm:output:0" 
transpose_4transpose_4_0:y:0"-
transpose_4_permtranspose_4/perm:output:0" 
transpose_5transpose_5_0:y:0"-
transpose_5_permtranspose_5/perm:output:0" 
transpose_6transpose_6_0:y:0"-
transpose_6_permtranspose_6/perm:output:0" 
transpose_7transpose_7_0:y:0"-
transpose_7_permtranspose_7/perm:output:0")
transpose_permtranspose/perm:output:0*\
_input_shapesK
I:??????????:?????????:Z:Z:Z*<
api_implements*(gru_8d3b7182-ad7a-489b-86fa-c30b709435ce*
api_preferred_deviceGPU*T
backward_function_name:8__inference___backward_gpu_gru_with_fallback_16915_17133*
go_backwards( *

time_major( :T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinit_h:FB

_output_shapes

:Z
 
_user_specified_namekernel:PL

_output_shapes

:Z
*
_user_specified_namerecurrent_kernel:D@

_output_shapes

:Z

_user_specified_namebias
?	
?
while_cond_15313
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice3
/while_while_cond_15313___redundant_placeholder03
/while_while_cond_15313___redundant_placeholder13
/while_while_cond_15313___redundant_placeholder23
/while_while_cond_15313___redundant_placeholder33
/while_while_cond_15313___redundant_placeholder4
while_identity
n

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*D
_input_shapes3
1: : : : :?????????: :::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
::

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
?
?
>__inference_gru_layer_call_and_return_conditional_losses_17135

inputs 
read_readvariableop_resource"
read_1_readvariableop_resource"
read_2_readvariableop_resource

identity_3??Read/ReadVariableOp?Read_1/ReadVariableOp?Read_2/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????2
zeros?
Read/ReadVariableOpReadVariableOpread_readvariableop_resource*
_output_shapes

:Z*
dtype02
Read/ReadVariableOpf
IdentityIdentityRead/ReadVariableOp:value:0*
T0*
_output_shapes

:Z2

Identity?
Read_1/ReadVariableOpReadVariableOpread_1_readvariableop_resource*
_output_shapes

:Z*
dtype02
Read_1/ReadVariableOpl

Identity_1IdentityRead_1/ReadVariableOp:value:0*
T0*
_output_shapes

:Z2

Identity_1?
Read_2/ReadVariableOpReadVariableOpread_2_readvariableop_resource*
_output_shapes

:Z*
dtype02
Read_2/ReadVariableOpl

Identity_2IdentityRead_2/ReadVariableOp:value:0*
T0*
_output_shapes

:Z2

Identity_2?
PartitionedCallPartitionedCallinputszeros:output:0Identity:output:0Identity_1:output:0Identity_2:output:0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *T
_output_shapesB
@:?????????:??????????:?????????: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *'
f"R 
__inference_standard_gru_168352
PartitionedCall?

Identity_3IdentityPartitionedCall:output:0^Read/ReadVariableOp^Read_1/ReadVariableOp^Read_2/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_3"!

identity_3Identity_3:output:0*7
_input_shapes&
$:??????????:::2*
Read/ReadVariableOpRead/ReadVariableOp2.
Read_1/ReadVariableOpRead_1/ReadVariableOp2.
Read_2/ReadVariableOpRead_2/ReadVariableOp:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?M
?
%__forward_gpu_gru_with_fallback_17763

inputs
init_h_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3
cudnnrnn
strided_slice
transpose_7_perm
transpose_7

cudnnrnn_0

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
cudnnrnn_input_c

concat
transpose_perm

init_h
concat_axis
	reshape_1
	reshape_2
	reshape_3
	reshape_4
	reshape_5
	reshape_6
	reshape_7
	reshape_8
	reshape_9

reshape_10

reshape_11

reshape_12
transpose_1_perm
transpose_1
transpose_2_perm
transpose_2
transpose_3_perm
transpose_3
transpose_4_perm
transpose_4
transpose_5_perm
transpose_5
transpose_6_perm
transpose_6
split_2_split_dim
split_split_dim	
split
split_1_split_dim
split_1
reshape?u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permO
transpose_0	Transposeinputstranspose/perm:output:0*
T02
	transposeb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims/dim

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2

ExpandDimsP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dimU
split_0Splitsplit/split_dim:output:0kernel*
T0*
	num_split2
splitT
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dime
	split_1_0Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*
	num_split2	
split_1q
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
Reshape/shapea
ReshapeReshapebiasReshape/shape:output:0*
T0*
_output_shapes	
:?2	
ReshapeT
Const_2Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_2h
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_2/split_dim?
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*8
_output_shapes&
$::::::*
	num_split2	
split_2e
Const_3Const*
_output_shapes
:*
dtype0*
valueB:
?????????2	
Const_3u
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_1/perm_
transpose_1_0	Transposesplit_0:output:0transpose_1/perm:output:0*
T02
transpose_1l
	Reshape_1Reshapetranspose_1_0:y:0Const_3:output:0*
T0*
_output_shapes	
:?2
	Reshape_1u
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_2/perm_
transpose_2_0	Transposesplit_0:output:1transpose_2/perm:output:0*
T02
transpose_2l
	Reshape_2Reshapetranspose_2_0:y:0Const_3:output:0*
T0*
_output_shapes	
:?2
	Reshape_2u
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_3/perm_
transpose_3_0	Transposesplit_0:output:2transpose_3/perm:output:0*
T02
transpose_3l
	Reshape_3Reshapetranspose_3_0:y:0Const_3:output:0*
T0*
_output_shapes	
:?2
	Reshape_3u
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_4/perma
transpose_4_0	Transposesplit_1_0:output:0transpose_4/perm:output:0*
T02
transpose_4l
	Reshape_4Reshapetranspose_4_0:y:0Const_3:output:0*
T0*
_output_shapes	
:?2
	Reshape_4u
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_5/perma
transpose_5_0	Transposesplit_1_0:output:1transpose_5/perm:output:0*
T02
transpose_5l
	Reshape_5Reshapetranspose_5_0:y:0Const_3:output:0*
T0*
_output_shapes	
:?2
	Reshape_5u
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_6/perma
transpose_6_0	Transposesplit_1_0:output:2transpose_6/perm:output:0*
T02
transpose_6l
	Reshape_6Reshapetranspose_6_0:y:0Const_3:output:0*
T0*
_output_shapes	
:?2
	Reshape_6j
	Reshape_7Reshapesplit_2:output:0Const_3:output:0*
T0*
_output_shapes
:2
	Reshape_7j
	Reshape_8Reshapesplit_2:output:1Const_3:output:0*
T0*
_output_shapes
:2
	Reshape_8j
	Reshape_9Reshapesplit_2:output:2Const_3:output:0*
T0*
_output_shapes
:2
	Reshape_9l

Reshape_10Reshapesplit_2:output:3Const_3:output:0*
T0*
_output_shapes
:2

Reshape_10l

Reshape_11Reshapesplit_2:output:4Const_3:output:0*
T0*
_output_shapes
:2

Reshape_11l

Reshape_12Reshapesplit_2:output:5Const_3:output:0*
T0*
_output_shapes
:2

Reshape_12\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat/axis?
concat_0ConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
N*
T02
concati
CudnnRNN/input_cConst*
_output_shapes
: *
dtype0*
valueB
 *    2
CudnnRNN/input_c?
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat_0:output:0*
T0*I
_output_shapes7
5:??????????:?????????: :*
rnn_modegru2

CudnnRNN}
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_slice_0StridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
shrink_axis_mask2
strided_slicey
transpose_7/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_7/perm`
transpose_7_0	TransposeCudnnRNN:output:0transpose_7/perm:output:0*
T02
transpose_7{
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:?????????*
squeeze_dims
 2	
Squeezef
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @2	
runtimel
IdentityIdentitystrided_slice_0:output:0*
T0*'
_output_shapes
:?????????2

Identityn

Identity_1Identitytranspose_7_0:y:0*
T0*,
_output_shapes
:??????????2

Identity_1h

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:?????????2

Identity_2W

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: 2

Identity_3"
concatconcat_0:output:0"#
concat_axisconcat/axis:output:0"
cudnnrnnCudnnRNN:output:0"!

cudnnrnn_0CudnnRNN:output_h:0"!

cudnnrnn_1CudnnRNN:output_c:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"-
cudnnrnn_input_cCudnnRNN/input_c:output:0"!

expanddimsExpandDims:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"
init_hinit_h_0"
reshapeReshape:output:0"
	reshape_1Reshape_1:output:0"!

reshape_10Reshape_10:output:0"!

reshape_11Reshape_11:output:0"!

reshape_12Reshape_12:output:0"
	reshape_2Reshape_2:output:0"
	reshape_3Reshape_3:output:0"
	reshape_4Reshape_4:output:0"
	reshape_5Reshape_5:output:0"
	reshape_6Reshape_6:output:0"
	reshape_7Reshape_7:output:0"
	reshape_8Reshape_8:output:0"
	reshape_9Reshape_9:output:0"
splitsplit_0:output:0"
split_1split_1_0:output:0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0")
strided_slicestrided_slice_0:output:0"
	transposetranspose_0:y:0" 
transpose_1transpose_1_0:y:0"-
transpose_1_permtranspose_1/perm:output:0" 
transpose_2transpose_2_0:y:0"-
transpose_2_permtranspose_2/perm:output:0" 
transpose_3transpose_3_0:y:0"-
transpose_3_permtranspose_3/perm:output:0" 
transpose_4transpose_4_0:y:0"-
transpose_4_permtranspose_4/perm:output:0" 
transpose_5transpose_5_0:y:0"-
transpose_5_permtranspose_5/perm:output:0" 
transpose_6transpose_6_0:y:0"-
transpose_6_permtranspose_6/perm:output:0" 
transpose_7transpose_7_0:y:0"-
transpose_7_permtranspose_7/perm:output:0")
transpose_permtranspose/perm:output:0*\
_input_shapesK
I:??????????:?????????:Z:Z:Z*<
api_implements*(gru_5f9dfa88-82d0-4c61-bd4c-33d58af38bad*
api_preferred_deviceGPU*T
backward_function_name:8__inference___backward_gpu_gru_with_fallback_17546_17764*
go_backwards( *

time_major( :T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinit_h:FB

_output_shapes

:Z
 
_user_specified_namekernel:PL

_output_shapes

:Z
*
_user_specified_namerecurrent_kernel:D@

_output_shapes

:Z

_user_specified_namebias
?M
?
%__forward_gpu_gru_with_fallback_18745

inputs
init_h_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3
cudnnrnn
strided_slice
transpose_7_perm
transpose_7

cudnnrnn_0

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
cudnnrnn_input_c

concat
transpose_perm

init_h
concat_axis
	reshape_1
	reshape_2
	reshape_3
	reshape_4
	reshape_5
	reshape_6
	reshape_7
	reshape_8
	reshape_9

reshape_10

reshape_11

reshape_12
transpose_1_perm
transpose_1
transpose_2_perm
transpose_2
transpose_3_perm
transpose_3
transpose_4_perm
transpose_4
transpose_5_perm
transpose_5
transpose_6_perm
transpose_6
split_2_split_dim
split_split_dim	
split
split_1_split_dim
split_1
reshape?u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permO
transpose_0	Transposeinputstranspose/perm:output:0*
T02
	transposeb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims/dim

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2

ExpandDimsP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dimU
split_0Splitsplit/split_dim:output:0kernel*
T0*
	num_split2
splitT
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dime
	split_1_0Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*
	num_split2	
split_1q
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
Reshape/shapea
ReshapeReshapebiasReshape/shape:output:0*
T0*
_output_shapes	
:?2	
ReshapeT
Const_2Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_2h
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_2/split_dim?
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*8
_output_shapes&
$::::::*
	num_split2	
split_2e
Const_3Const*
_output_shapes
:*
dtype0*
valueB:
?????????2	
Const_3u
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_1/perm_
transpose_1_0	Transposesplit_0:output:0transpose_1/perm:output:0*
T02
transpose_1l
	Reshape_1Reshapetranspose_1_0:y:0Const_3:output:0*
T0*
_output_shapes	
:?2
	Reshape_1u
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_2/perm_
transpose_2_0	Transposesplit_0:output:1transpose_2/perm:output:0*
T02
transpose_2l
	Reshape_2Reshapetranspose_2_0:y:0Const_3:output:0*
T0*
_output_shapes	
:?2
	Reshape_2u
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_3/perm_
transpose_3_0	Transposesplit_0:output:2transpose_3/perm:output:0*
T02
transpose_3l
	Reshape_3Reshapetranspose_3_0:y:0Const_3:output:0*
T0*
_output_shapes	
:?2
	Reshape_3u
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_4/perma
transpose_4_0	Transposesplit_1_0:output:0transpose_4/perm:output:0*
T02
transpose_4l
	Reshape_4Reshapetranspose_4_0:y:0Const_3:output:0*
T0*
_output_shapes	
:?2
	Reshape_4u
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_5/perma
transpose_5_0	Transposesplit_1_0:output:1transpose_5/perm:output:0*
T02
transpose_5l
	Reshape_5Reshapetranspose_5_0:y:0Const_3:output:0*
T0*
_output_shapes	
:?2
	Reshape_5u
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_6/perma
transpose_6_0	Transposesplit_1_0:output:2transpose_6/perm:output:0*
T02
transpose_6l
	Reshape_6Reshapetranspose_6_0:y:0Const_3:output:0*
T0*
_output_shapes	
:?2
	Reshape_6j
	Reshape_7Reshapesplit_2:output:0Const_3:output:0*
T0*
_output_shapes
:2
	Reshape_7j
	Reshape_8Reshapesplit_2:output:1Const_3:output:0*
T0*
_output_shapes
:2
	Reshape_8j
	Reshape_9Reshapesplit_2:output:2Const_3:output:0*
T0*
_output_shapes
:2
	Reshape_9l

Reshape_10Reshapesplit_2:output:3Const_3:output:0*
T0*
_output_shapes
:2

Reshape_10l

Reshape_11Reshapesplit_2:output:4Const_3:output:0*
T0*
_output_shapes
:2

Reshape_11l

Reshape_12Reshapesplit_2:output:5Const_3:output:0*
T0*
_output_shapes
:2

Reshape_12\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat/axis?
concat_0ConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
N*
T02
concati
CudnnRNN/input_cConst*
_output_shapes
: *
dtype0*
valueB
 *    2
CudnnRNN/input_c?
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat_0:output:0*
T0*Q
_output_shapes?
=:??????????????????:?????????: :*
rnn_modegru2

CudnnRNN}
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_slice_0StridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
shrink_axis_mask2
strided_slicey
transpose_7/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_7/perm`
transpose_7_0	TransposeCudnnRNN:output:0transpose_7/perm:output:0*
T02
transpose_7{
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:?????????*
squeeze_dims
 2	
Squeezef
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @2	
runtimel
IdentityIdentitystrided_slice_0:output:0*
T0*'
_output_shapes
:?????????2

Identityv

Identity_1Identitytranspose_7_0:y:0*
T0*4
_output_shapes"
 :??????????????????2

Identity_1h

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:?????????2

Identity_2W

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: 2

Identity_3"
concatconcat_0:output:0"#
concat_axisconcat/axis:output:0"
cudnnrnnCudnnRNN:output:0"!

cudnnrnn_0CudnnRNN:output_h:0"!

cudnnrnn_1CudnnRNN:output_c:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"-
cudnnrnn_input_cCudnnRNN/input_c:output:0"!

expanddimsExpandDims:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"
init_hinit_h_0"
reshapeReshape:output:0"
	reshape_1Reshape_1:output:0"!

reshape_10Reshape_10:output:0"!

reshape_11Reshape_11:output:0"!

reshape_12Reshape_12:output:0"
	reshape_2Reshape_2:output:0"
	reshape_3Reshape_3:output:0"
	reshape_4Reshape_4:output:0"
	reshape_5Reshape_5:output:0"
	reshape_6Reshape_6:output:0"
	reshape_7Reshape_7:output:0"
	reshape_8Reshape_8:output:0"
	reshape_9Reshape_9:output:0"
splitsplit_0:output:0"
split_1split_1_0:output:0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0")
strided_slicestrided_slice_0:output:0"
	transposetranspose_0:y:0" 
transpose_1transpose_1_0:y:0"-
transpose_1_permtranspose_1/perm:output:0" 
transpose_2transpose_2_0:y:0"-
transpose_2_permtranspose_2/perm:output:0" 
transpose_3transpose_3_0:y:0"-
transpose_3_permtranspose_3/perm:output:0" 
transpose_4transpose_4_0:y:0"-
transpose_4_permtranspose_4/perm:output:0" 
transpose_5transpose_5_0:y:0"-
transpose_5_permtranspose_5/perm:output:0" 
transpose_6transpose_6_0:y:0"-
transpose_6_permtranspose_6/perm:output:0" 
transpose_7transpose_7_0:y:0"-
transpose_7_permtranspose_7/perm:output:0")
transpose_permtranspose/perm:output:0*d
_input_shapesS
Q:??????????????????:?????????:Z:Z:Z*<
api_implements*(gru_7fd55799-5a12-43e2-bc93-7077640e6183*
api_preferred_deviceGPU*T
backward_function_name:8__inference___backward_gpu_gru_with_fallback_18528_18746*
go_backwards( *

time_major( :\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinit_h:FB

_output_shapes

:Z
 
_user_specified_namekernel:PL

_output_shapes

:Z
*
_user_specified_namerecurrent_kernel:D@

_output_shapes

:Z

_user_specified_namebias
?	
?
while_cond_17850
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice3
/while_while_cond_17850___redundant_placeholder03
/while_while_cond_17850___redundant_placeholder13
/while_while_cond_17850___redundant_placeholder23
/while_while_cond_17850___redundant_placeholder33
/while_while_cond_17850___redundant_placeholder4
while_identity
n

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*D
_input_shapes3
1: : : : :?????????: :::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
::

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
?
?
#__inference_gru_layer_call_fn_19228
inputs_0
unknown
	unknown_0
	unknown_1
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *G
fBR@
>__inference_gru_layer_call_and_return_conditional_losses_157052
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????:::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :??????????????????
"
_user_specified_name
inputs/0
?	
?
while_cond_18825
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice3
/while_while_cond_18825___redundant_placeholder03
/while_while_cond_18825___redundant_placeholder13
/while_while_cond_18825___redundant_placeholder23
/while_while_cond_18825___redundant_placeholder33
/while_while_cond_18825___redundant_placeholder4
while_identity
n

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*D
_input_shapes3
1: : : : :?????????: :::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
::

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
?E
?
__inference_standard_gru_18917

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3X
unstackUnpackbias*
T0* 
_output_shapes
:Z:Z*	
num2	
unstacku
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????2
	transposeK
ShapeShapetranspose:y:0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_1n
MatMulMatMulstrided_slice_1:output:0kernel*
T0*'
_output_shapes
:?????????Z2
MatMuls
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*'
_output_shapes
:?????????Z2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:?????????:?????????:?????????*
	num_split2
splitj
MatMul_1MatMulinit_hrecurrent_kernel*
T0*'
_output_shapes
:?????????Z2

MatMul_1y
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*'
_output_shapes
:?????????Z2
	BiasAdd_1T
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dim?
split_1Splitsplit_1/split_dim:output:0BiasAdd_1:output:0*
T0*M
_output_shapes;
9:?????????:?????????:?????????*
	num_split2	
split_1g
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:?????????2
addX
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:?????????2	
Sigmoidk
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:?????????2
add_1^
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:?????????2
	Sigmoid_1d
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:?????????2
mulb
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:?????????2
add_2Q
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:?????????2
Tanh\
mul_1MulSigmoid:y:0init_h*
T0*'
_output_shapes
:?????????2
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
sub/x`
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:?????????2
subZ
mul_2Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:?????????2
mul_2_
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:?????????2
add_3?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileStatelessWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelunstack:output:0recurrent_kernelunstack:output:1*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*S
_output_shapesA
?: : : : :?????????: : :Z:Z:Z:Z* 
_read_only_resource_inputs
 *
bodyR
while_body_18826*
condR
while_cond_18825*R
output_shapesA
?: : : : :?????????: : :Z:Z:Z:Z*
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_2y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  ??2	
runtimel
IdentityIdentitystrided_slice_2:output:0*
T0*'
_output_shapes
:?????????2

Identityt

Identity_1Identitytranspose_1:y:0*
T0*4
_output_shapes"
 :??????????????????2

Identity_1f

Identity_2Identitywhile:output:4*
T0*'
_output_shapes
:?????????2

Identity_2W

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: 2

Identity_3"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*d
_input_shapesS
Q:??????????????????:?????????:Z:Z:Z*<
api_implements*(gru_d29f5daa-115b-49e1-809f-181a86e00a5d*
api_preferred_deviceCPU*
go_backwards( *

time_major( :\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinit_h:FB

_output_shapes

:Z
 
_user_specified_namekernel:PL

_output_shapes

:Z
*
_user_specified_namerecurrent_kernel:D@

_output_shapes

:Z

_user_specified_namebias
?:
?

__inference__traced_save_20320
file_prefix-
)savev2_dense_2_kernel_read_readvariableop+
'savev2_dense_2_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop2
.savev2_gru_gru_cell_kernel_read_readvariableop<
8savev2_gru_gru_cell_recurrent_kernel_read_readvariableop0
,savev2_gru_gru_cell_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop&
"savev2_total_2_read_readvariableop&
"savev2_count_2_read_readvariableop4
0savev2_adam_dense_2_kernel_m_read_readvariableop2
.savev2_adam_dense_2_bias_m_read_readvariableop9
5savev2_adam_gru_gru_cell_kernel_m_read_readvariableopC
?savev2_adam_gru_gru_cell_recurrent_kernel_m_read_readvariableop7
3savev2_adam_gru_gru_cell_bias_m_read_readvariableop4
0savev2_adam_dense_2_kernel_v_read_readvariableop2
.savev2_adam_dense_2_bias_v_read_readvariableop9
5savev2_adam_gru_gru_cell_kernel_v_read_readvariableopC
?savev2_adam_gru_gru_cell_recurrent_kernel_v_read_readvariableop7
3savev2_adam_gru_gru_cell_bias_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*I
value@B>B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?

SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0)savev2_dense_2_kernel_read_readvariableop'savev2_dense_2_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop.savev2_gru_gru_cell_kernel_read_readvariableop8savev2_gru_gru_cell_recurrent_kernel_read_readvariableop,savev2_gru_gru_cell_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop"savev2_total_2_read_readvariableop"savev2_count_2_read_readvariableop0savev2_adam_dense_2_kernel_m_read_readvariableop.savev2_adam_dense_2_bias_m_read_readvariableop5savev2_adam_gru_gru_cell_kernel_m_read_readvariableop?savev2_adam_gru_gru_cell_recurrent_kernel_m_read_readvariableop3savev2_adam_gru_gru_cell_bias_m_read_readvariableop0savev2_adam_dense_2_kernel_v_read_readvariableop.savev2_adam_dense_2_bias_v_read_readvariableop5savev2_adam_gru_gru_cell_kernel_v_read_readvariableop?savev2_adam_gru_gru_cell_recurrent_kernel_v_read_readvariableop3savev2_adam_gru_gru_cell_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *)
dtypes
2	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*?
_input_shapes?
?: :	?:?: : : : : :Z:Z:Z: : : : : : :	?:?:Z:Z:Z:	?:?:Z:Z:Z: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	?:!

_output_shapes	
:?:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:Z:$	 

_output_shapes

:Z:$
 

_output_shapes

:Z:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	?:!

_output_shapes	
:?:$ 

_output_shapes

:Z:$ 

_output_shapes

:Z:$ 

_output_shapes

:Z:%!

_output_shapes
:	?:!

_output_shapes	
:?:$ 

_output_shapes

:Z:$ 

_output_shapes

:Z:$ 

_output_shapes

:Z:

_output_shapes
: "?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
D
	gru_input7
serving_default_gru_input:0??????????<
dense_21
StatefulPartitionedCall:0??????????tensorflow/serving/predict:ؖ
?#
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api

signatures
Q__call__
R_default_save_signature
*S&call_and_return_all_conditional_losses"?!
_tf_keras_sequential?!{"class_name": "Sequential", "name": "sequential_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 240, 5]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "gru_input"}}, {"class_name": "GRU", "config": {"name": "gru", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 240, 5]}, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 30, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2, "reset_after": true}}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 180, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 5]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 240, 5]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 240, 5]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "gru_input"}}, {"class_name": "GRU", "config": {"name": "gru", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 240, 5]}, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 30, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2, "reset_after": true}}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 180, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": "mean_squared_error", "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "mae", "dtype": "float32", "fn": "mean_absolute_error"}}, {"class_name": "MeanMetricWrapper", "config": {"name": "acc", "dtype": "float32", "fn": "categorical_accuracy"}}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?
	cell


state_spec
	variables
trainable_variables
regularization_losses
	keras_api
T__call__
*U&call_and_return_all_conditional_losses"?
_tf_keras_rnn_layer?
{"class_name": "GRU", "name": "gru", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 240, 5]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "gru", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 240, 5]}, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 30, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2, "reset_after": true}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 5]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 240, 5]}}
?

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
V__call__
*W&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 180, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 30}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 30]}}
?
iter

beta_1

beta_2
	decay
learning_ratemGmHmImJmKvLvMvNvOvP"
	optimizer
C
0
1
2
3
4"
trackable_list_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
 "
trackable_list_wrapper
?
layer_metrics
	variables
trainable_variables
metrics
regularization_losses

layers
 non_trainable_variables
!layer_regularization_losses
Q__call__
R_default_save_signature
*S&call_and_return_all_conditional_losses
&S"call_and_return_conditional_losses"
_generic_user_object
,
Xserving_default"
signature_map
?

kernel
recurrent_kernel
bias
"	variables
#trainable_variables
$regularization_losses
%	keras_api
Y__call__
*Z&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "GRUCell", "name": "gru_cell", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "gru_cell", "trainable": true, "dtype": "float32", "units": 30, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2, "reset_after": true}}
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
?
&layer_metrics
	variables

'states
trainable_variables
(metrics
regularization_losses

)layers
*non_trainable_variables
+layer_regularization_losses
T__call__
*U&call_and_return_all_conditional_losses
&U"call_and_return_conditional_losses"
_generic_user_object
!:	?2dense_2/kernel
:?2dense_2/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
,layer_metrics
	variables
trainable_variables
-metrics
regularization_losses

.layers
/non_trainable_variables
0layer_regularization_losses
V__call__
*W&call_and_return_all_conditional_losses
&W"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
%:#Z2gru/gru_cell/kernel
/:-Z2gru/gru_cell/recurrent_kernel
#:!Z2gru/gru_cell/bias
 "
trackable_dict_wrapper
5
10
21
32"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
?
4layer_metrics
"	variables
#trainable_variables
5metrics
$regularization_losses

6layers
7non_trainable_variables
8layer_regularization_losses
Y__call__
*Z&call_and_return_all_conditional_losses
&Z"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
	0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
	9total
	:count
;	variables
<	keras_api"?
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
?
	=total
	>count
?
_fn_kwargs
@	variables
A	keras_api"?
_tf_keras_metric?{"class_name": "MeanMetricWrapper", "name": "mae", "dtype": "float32", "config": {"name": "mae", "dtype": "float32", "fn": "mean_absolute_error"}}
?
	Btotal
	Ccount
D
_fn_kwargs
E	variables
F	keras_api"?
_tf_keras_metric?{"class_name": "MeanMetricWrapper", "name": "acc", "dtype": "float32", "config": {"name": "acc", "dtype": "float32", "fn": "categorical_accuracy"}}
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:  (2total
:  (2count
.
90
:1"
trackable_list_wrapper
-
;	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
=0
>1"
trackable_list_wrapper
-
@	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
B0
C1"
trackable_list_wrapper
-
E	variables"
_generic_user_object
&:$	?2Adam/dense_2/kernel/m
 :?2Adam/dense_2/bias/m
*:(Z2Adam/gru/gru_cell/kernel/m
4:2Z2$Adam/gru/gru_cell/recurrent_kernel/m
(:&Z2Adam/gru/gru_cell/bias/m
&:$	?2Adam/dense_2/kernel/v
 :?2Adam/dense_2/bias/v
*:(Z2Adam/gru/gru_cell/kernel/v
4:2Z2$Adam/gru/gru_cell/recurrent_kernel/v
(:&Z2Adam/gru/gru_cell/bias/v
?2?
,__inference_sequential_1_layer_call_fn_18279
,__inference_sequential_1_layer_call_fn_18264
,__inference_sequential_1_layer_call_fn_17272
,__inference_sequential_1_layer_call_fn_17241?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
 __inference__wrapped_model_14294?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *-?*
(?%
	gru_input??????????
?2?
G__inference_sequential_1_layer_call_and_return_conditional_losses_18249
G__inference_sequential_1_layer_call_and_return_conditional_losses_17773
G__inference_sequential_1_layer_call_and_return_conditional_losses_17209
G__inference_sequential_1_layer_call_and_return_conditional_losses_17193?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
#__inference_gru_layer_call_fn_20188
#__inference_gru_layer_call_fn_20199
#__inference_gru_layer_call_fn_19239
#__inference_gru_layer_call_fn_19228?
???
FullArgSpecB
args:?7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults?

 
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
>__inference_gru_layer_call_and_return_conditional_losses_20177
>__inference_gru_layer_call_and_return_conditional_losses_18748
>__inference_gru_layer_call_and_return_conditional_losses_19217
>__inference_gru_layer_call_and_return_conditional_losses_19708?
???
FullArgSpecB
args:?7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults?

 
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
'__inference_dense_2_layer_call_fn_20219?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_dense_2_layer_call_and_return_conditional_losses_20210?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
#__inference_signature_wrapper_17297	gru_input"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec3
args+?(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpec3
args+?(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 ?
 __inference__wrapped_model_14294t7?4
-?*
(?%
	gru_input??????????
? "2?/
-
dense_2"?
dense_2???????????
B__inference_dense_2_layer_call_and_return_conditional_losses_20210]/?,
%?"
 ?
inputs?????????
? "&?#
?
0??????????
? {
'__inference_dense_2_layer_call_fn_20219P/?,
%?"
 ?
inputs?????????
? "????????????
>__inference_gru_layer_call_and_return_conditional_losses_18748}O?L
E?B
4?1
/?,
inputs/0??????????????????

 
p

 
? "%?"
?
0?????????
? ?
>__inference_gru_layer_call_and_return_conditional_losses_19217}O?L
E?B
4?1
/?,
inputs/0??????????????????

 
p 

 
? "%?"
?
0?????????
? ?
>__inference_gru_layer_call_and_return_conditional_losses_19708n@?=
6?3
%?"
inputs??????????

 
p

 
? "%?"
?
0?????????
? ?
>__inference_gru_layer_call_and_return_conditional_losses_20177n@?=
6?3
%?"
inputs??????????

 
p 

 
? "%?"
?
0?????????
? ?
#__inference_gru_layer_call_fn_19228pO?L
E?B
4?1
/?,
inputs/0??????????????????

 
p

 
? "???????????
#__inference_gru_layer_call_fn_19239pO?L
E?B
4?1
/?,
inputs/0??????????????????

 
p 

 
? "???????????
#__inference_gru_layer_call_fn_20188a@?=
6?3
%?"
inputs??????????

 
p

 
? "???????????
#__inference_gru_layer_call_fn_20199a@?=
6?3
%?"
inputs??????????

 
p 

 
? "???????????
G__inference_sequential_1_layer_call_and_return_conditional_losses_17193p??<
5?2
(?%
	gru_input??????????
p

 
? "&?#
?
0??????????
? ?
G__inference_sequential_1_layer_call_and_return_conditional_losses_17209p??<
5?2
(?%
	gru_input??????????
p 

 
? "&?#
?
0??????????
? ?
G__inference_sequential_1_layer_call_and_return_conditional_losses_17773m<?9
2?/
%?"
inputs??????????
p

 
? "&?#
?
0??????????
? ?
G__inference_sequential_1_layer_call_and_return_conditional_losses_18249m<?9
2?/
%?"
inputs??????????
p 

 
? "&?#
?
0??????????
? ?
,__inference_sequential_1_layer_call_fn_17241c??<
5?2
(?%
	gru_input??????????
p

 
? "????????????
,__inference_sequential_1_layer_call_fn_17272c??<
5?2
(?%
	gru_input??????????
p 

 
? "????????????
,__inference_sequential_1_layer_call_fn_18264`<?9
2?/
%?"
inputs??????????
p

 
? "????????????
,__inference_sequential_1_layer_call_fn_18279`<?9
2?/
%?"
inputs??????????
p 

 
? "????????????
#__inference_signature_wrapper_17297?D?A
? 
:?7
5
	gru_input(?%
	gru_input??????????"2?/
-
dense_2"?
dense_2??????????