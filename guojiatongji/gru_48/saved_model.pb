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
z
dense_10/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:0* 
shared_namedense_10/kernel
s
#dense_10/kernel/Read/ReadVariableOpReadVariableOpdense_10/kernel*
_output_shapes

:0*
dtype0
r
dense_10/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*
shared_namedense_10/bias
k
!dense_10/bias/Read/ReadVariableOpReadVariableOpdense_10/bias*
_output_shapes
:0*
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
gru_1/gru_cell_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:Z*(
shared_namegru_1/gru_cell_2/kernel
?
+gru_1/gru_cell_2/kernel/Read/ReadVariableOpReadVariableOpgru_1/gru_cell_2/kernel*
_output_shapes

:Z*
dtype0
?
!gru_1/gru_cell_2/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:Z*2
shared_name#!gru_1/gru_cell_2/recurrent_kernel
?
5gru_1/gru_cell_2/recurrent_kernel/Read/ReadVariableOpReadVariableOp!gru_1/gru_cell_2/recurrent_kernel*
_output_shapes

:Z*
dtype0
?
gru_1/gru_cell_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:Z*&
shared_namegru_1/gru_cell_2/bias

)gru_1/gru_cell_2/bias/Read/ReadVariableOpReadVariableOpgru_1/gru_cell_2/bias*
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
Adam/dense_10/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:0*'
shared_nameAdam/dense_10/kernel/m
?
*Adam/dense_10/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_10/kernel/m*
_output_shapes

:0*
dtype0
?
Adam/dense_10/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*%
shared_nameAdam/dense_10/bias/m
y
(Adam/dense_10/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_10/bias/m*
_output_shapes
:0*
dtype0
?
Adam/gru_1/gru_cell_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:Z*/
shared_name Adam/gru_1/gru_cell_2/kernel/m
?
2Adam/gru_1/gru_cell_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/gru_1/gru_cell_2/kernel/m*
_output_shapes

:Z*
dtype0
?
(Adam/gru_1/gru_cell_2/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:Z*9
shared_name*(Adam/gru_1/gru_cell_2/recurrent_kernel/m
?
<Adam/gru_1/gru_cell_2/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp(Adam/gru_1/gru_cell_2/recurrent_kernel/m*
_output_shapes

:Z*
dtype0
?
Adam/gru_1/gru_cell_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:Z*-
shared_nameAdam/gru_1/gru_cell_2/bias/m
?
0Adam/gru_1/gru_cell_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/gru_1/gru_cell_2/bias/m*
_output_shapes

:Z*
dtype0
?
Adam/dense_10/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:0*'
shared_nameAdam/dense_10/kernel/v
?
*Adam/dense_10/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_10/kernel/v*
_output_shapes

:0*
dtype0
?
Adam/dense_10/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*%
shared_nameAdam/dense_10/bias/v
y
(Adam/dense_10/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_10/bias/v*
_output_shapes
:0*
dtype0
?
Adam/gru_1/gru_cell_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:Z*/
shared_name Adam/gru_1/gru_cell_2/kernel/v
?
2Adam/gru_1/gru_cell_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/gru_1/gru_cell_2/kernel/v*
_output_shapes

:Z*
dtype0
?
(Adam/gru_1/gru_cell_2/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:Z*9
shared_name*(Adam/gru_1/gru_cell_2/recurrent_kernel/v
?
<Adam/gru_1/gru_cell_2/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp(Adam/gru_1/gru_cell_2/recurrent_kernel/v*
_output_shapes

:Z*
dtype0
?
Adam/gru_1/gru_cell_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:Z*-
shared_nameAdam/gru_1/gru_cell_2/bias/v
?
0Adam/gru_1/gru_cell_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/gru_1/gru_cell_2/bias/v*
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
regularization_losses
trainable_variables
	keras_api

signatures
l
	cell


state_spec
	variables
regularization_losses
trainable_variables
	keras_api
h

kernel
bias
	variables
regularization_losses
trainable_variables
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
 
#
0
1
2
3
4
?

layers
	variables
metrics
regularization_losses
layer_metrics
trainable_variables
 non_trainable_variables
!layer_regularization_losses
 
~

kernel
recurrent_kernel
bias
"	variables
#regularization_losses
$trainable_variables
%	keras_api
 

0
1
2
 

0
1
2
?

&layers

'states
	variables
(metrics
regularization_losses
)layer_metrics
trainable_variables
*non_trainable_variables
+layer_regularization_losses
[Y
VARIABLE_VALUEdense_10/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_10/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
?

,layers
	variables
-metrics
regularization_losses
.layer_metrics
trainable_variables
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
SQ
VARIABLE_VALUEgru_1/gru_cell_2/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUE!gru_1/gru_cell_2/recurrent_kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEgru_1/gru_cell_2/bias&variables/2/.ATTRIBUTES/VARIABLE_VALUE

0
1

10
21
32
 
 
 

0
1
2
 

0
1
2
?

4layers
"	variables
5metrics
#regularization_losses
6layer_metrics
$trainable_variables
7non_trainable_variables
8layer_regularization_losses

	0
 
 
 
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
~|
VARIABLE_VALUEAdam/dense_10/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_10/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/gru_1/gru_cell_2/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUE(Adam/gru_1/gru_cell_2/recurrent_kernel/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUEAdam/gru_1/gru_cell_2/bias/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_10/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_10/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/gru_1/gru_cell_2/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUE(Adam/gru_1/gru_cell_2/recurrent_kernel/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUEAdam/gru_1/gru_cell_2/bias/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_gru_1_inputPlaceholder*+
_output_shapes
:?????????`*
dtype0* 
shape:?????????`
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_gru_1_inputgru_1/gru_cell_2/kernel!gru_1/gru_cell_2/recurrent_kernelgru_1/gru_cell_2/biasdense_10/kerneldense_10/bias*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????0*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8? *,
f'R%
#__inference_signature_wrapper_84044
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_10/kernel/Read/ReadVariableOp!dense_10/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp+gru_1/gru_cell_2/kernel/Read/ReadVariableOp5gru_1/gru_cell_2/recurrent_kernel/Read/ReadVariableOp)gru_1/gru_cell_2/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal_2/Read/ReadVariableOpcount_2/Read/ReadVariableOp*Adam/dense_10/kernel/m/Read/ReadVariableOp(Adam/dense_10/bias/m/Read/ReadVariableOp2Adam/gru_1/gru_cell_2/kernel/m/Read/ReadVariableOp<Adam/gru_1/gru_cell_2/recurrent_kernel/m/Read/ReadVariableOp0Adam/gru_1/gru_cell_2/bias/m/Read/ReadVariableOp*Adam/dense_10/kernel/v/Read/ReadVariableOp(Adam/dense_10/bias/v/Read/ReadVariableOp2Adam/gru_1/gru_cell_2/kernel/v/Read/ReadVariableOp<Adam/gru_1/gru_cell_2/recurrent_kernel/v/Read/ReadVariableOp0Adam/gru_1/gru_cell_2/bias/v/Read/ReadVariableOpConst*'
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
__inference__traced_save_87067
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_10/kerneldense_10/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rategru_1/gru_cell_2/kernel!gru_1/gru_cell_2/recurrent_kernelgru_1/gru_cell_2/biastotalcounttotal_1count_1total_2count_2Adam/dense_10/kernel/mAdam/dense_10/bias/mAdam/gru_1/gru_cell_2/kernel/m(Adam/gru_1/gru_cell_2/recurrent_kernel/mAdam/gru_1/gru_cell_2/bias/mAdam/dense_10/kernel/vAdam/dense_10/bias/vAdam/gru_1/gru_cell_2/kernel/v(Adam/gru_1/gru_cell_2/recurrent_kernel/vAdam/gru_1/gru_cell_2/bias/v*&
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
!__inference__traced_restore_87155??6
?	
?
while_cond_80642
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice3
/while_while_cond_80642___redundant_placeholder03
/while_while_cond_80642___redundant_placeholder13
/while_while_cond_80642___redundant_placeholder23
/while_while_cond_80642___redundant_placeholder33
/while_while_cond_80642___redundant_placeholder4
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
?
%__inference_gru_1_layer_call_fn_85975

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
GPU 2J 8? *I
fDRB
@__inference_gru_1_layer_call_and_return_conditional_losses_834132
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????`:::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????`
 
_user_specified_nameinputs
?
?
,__inference_sequential_8_layer_call_fn_83988
gru_1_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallgru_1_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????0*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_sequential_8_layer_call_and_return_conditional_losses_839752
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????02

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????`:::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
+
_output_shapes
:?????????`
%
_user_specified_namegru_1_input
?	
?
while_cond_84597
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice3
/while_while_cond_84597___redundant_placeholder03
/while_while_cond_84597___redundant_placeholder13
/while_while_cond_84597___redundant_placeholder23
/while_while_cond_84597___redundant_placeholder33
/while_while_cond_84597___redundant_placeholder4
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
?	
?
while_cond_84121
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice3
/while_while_cond_84121___redundant_placeholder03
/while_while_cond_84121___redundant_placeholder13
/while_while_cond_84121___redundant_placeholder23
/while_while_cond_84121___redundant_placeholder33
/while_while_cond_84121___redundant_placeholder4
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
?=
?
'__inference_gpu_gru_with_fallback_86703

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
api_implements*(gru_8264b2e4-d807-4ce4-88e8-68c6457615e8*
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
while_body_80643
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
while_cond_83021
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice3
/while_while_cond_83021___redundant_placeholder03
/while_while_cond_83021___redundant_placeholder13
/while_while_cond_83021___redundant_placeholder23
/while_while_cond_83021___redundant_placeholder33
/while_while_cond_83021___redundant_placeholder4
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
__inference_standard_gru_82152

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
while_body_82061*
condR
while_cond_82060*R
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
api_implements*(gru_4ee19f71-0a49-4281-adb9-3b2ff34755fa*
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
?2
?
while_body_86533
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
?M
?
%__forward_gpu_gru_with_fallback_84510

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
T0*H
_output_shapes6
4:`?????????:?????????: :*
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

Identitym

Identity_1Identitytranspose_7_0:y:0*
T0*+
_output_shapes
:?????????`2

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
transpose_permtranspose/perm:output:0*[
_input_shapesJ
H:?????????`:?????????:Z:Z:Z*<
api_implements*(gru_b643a1c6-e6ee-4f24-b167-0aaa19385eef*
api_preferred_deviceGPU*T
backward_function_name:8__inference___backward_gpu_gru_with_fallback_84293_84511*
go_backwards( *

time_major( :S O
+
_output_shapes
:?????????`
 
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
__inference_standard_gru_86155

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
while_body_86064*
condR
while_cond_86063*R
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
api_implements*(gru_dbdfb1bc-8350-49bc-99d4-4927254488fc*
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
__inference_standard_gru_82632

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
while_body_82541*
condR
while_cond_82540*R
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
api_implements*(gru_696f6c69-2d92-41ae-9f9b-48ad5b13a6ba*
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
?	
?
while_cond_82060
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice3
/while_while_cond_82060___redundant_placeholder03
/while_while_cond_82060___redundant_placeholder13
/while_while_cond_82060___redundant_placeholder23
/while_while_cond_82060___redundant_placeholder33
/while_while_cond_82060___redundant_placeholder4
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
?M
?
%__forward_gpu_gru_with_fallback_86921

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
api_implements*(gru_8264b2e4-d807-4ce4-88e8-68c6457615e8*
api_preferred_deviceGPU*T
backward_function_name:8__inference___backward_gpu_gru_with_fallback_86704_86922*
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
while_body_82541
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
'__inference_gpu_gru_with_fallback_84292

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
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:`?????????2
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
T0*H
_output_shapes6
4:`?????????:?????????: :*
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
T0*+
_output_shapes
:?????????`2
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

Identityk

Identity_1Identitytranspose_7:y:0*
T0*+
_output_shapes
:?????????`2

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

identity_3Identity_3:output:0*[
_input_shapesJ
H:?????????`:?????????:Z:Z:Z*<
api_implements*(gru_b643a1c6-e6ee-4f24-b167-0aaa19385eef*
api_preferred_deviceGPU*
go_backwards( *

time_major( :S O
+
_output_shapes
:?????????`
 
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
'__inference_gpu_gru_with_fallback_83661

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
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:`?????????2
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
T0*H
_output_shapes6
4:`?????????:?????????: :*
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
T0*+
_output_shapes
:?????????`2
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

Identityk

Identity_1Identitytranspose_7:y:0*
T0*+
_output_shapes
:?????????`2

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

identity_3Identity_3:output:0*[
_input_shapesJ
H:?????????`:?????????:Z:Z:Z*<
api_implements*(gru_4060a475-f519-4bbc-82de-47b15ccf83f2*
api_preferred_deviceGPU*
go_backwards( *

time_major( :S O
+
_output_shapes
:?????????`
 
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
while_cond_85103
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice3
/while_while_cond_85103___redundant_placeholder03
/while_while_cond_85103___redundant_placeholder13
/while_while_cond_85103___redundant_placeholder23
/while_while_cond_85103___redundant_placeholder33
/while_while_cond_85103___redundant_placeholder4
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
8__inference___backward_gpu_gru_with_fallback_82232_82450
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
api_implements*(gru_4ee19f71-0a49-4281-adb9-3b2ff34755fa*
api_preferred_deviceGPU*@
forward_function_name'%__forward_gpu_gru_with_fallback_82449*
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
?E
?
__inference_standard_gru_86624

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
while_body_86533*
condR
while_cond_86532*R
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
api_implements*(gru_8264b2e4-d807-4ce4-88e8-68c6457615e8*
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
?=
?
'__inference_gpu_gru_with_fallback_82231

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
api_implements*(gru_4ee19f71-0a49-4281-adb9-3b2ff34755fa*
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
?
?
%__inference_gru_1_layer_call_fn_86935
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
GPU 2J 8? *I
fDRB
@__inference_gru_1_layer_call_and_return_conditional_losses_824522
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
?'
?
G__inference_sequential_8_layer_call_and_return_conditional_losses_84520

inputs&
"gru_1_read_readvariableop_resource(
$gru_1_read_1_readvariableop_resource(
$gru_1_read_2_readvariableop_resource.
*dense_10_mlcmatmul_readvariableop_resource,
(dense_10_biasadd_readvariableop_resource
identity??dense_10/BiasAdd/ReadVariableOp?!dense_10/MLCMatMul/ReadVariableOp?gru_1/Read/ReadVariableOp?gru_1/Read_1/ReadVariableOp?gru_1/Read_2/ReadVariableOpP
gru_1/ShapeShapeinputs*
T0*
_output_shapes
:2
gru_1/Shape?
gru_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru_1/strided_slice/stack?
gru_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
gru_1/strided_slice/stack_1?
gru_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
gru_1/strided_slice/stack_2?
gru_1/strided_sliceStridedSlicegru_1/Shape:output:0"gru_1/strided_slice/stack:output:0$gru_1/strided_slice/stack_1:output:0$gru_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
gru_1/strided_sliceh
gru_1/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
gru_1/zeros/mul/y?
gru_1/zeros/mulMulgru_1/strided_slice:output:0gru_1/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
gru_1/zeros/mulk
gru_1/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
gru_1/zeros/Less/y
gru_1/zeros/LessLessgru_1/zeros/mul:z:0gru_1/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
gru_1/zeros/Lessn
gru_1/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
gru_1/zeros/packed/1?
gru_1/zeros/packedPackgru_1/strided_slice:output:0gru_1/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
gru_1/zeros/packedk
gru_1/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
gru_1/zeros/Const?
gru_1/zerosFillgru_1/zeros/packed:output:0gru_1/zeros/Const:output:0*
T0*'
_output_shapes
:?????????2
gru_1/zeros?
gru_1/Read/ReadVariableOpReadVariableOp"gru_1_read_readvariableop_resource*
_output_shapes

:Z*
dtype02
gru_1/Read/ReadVariableOpx
gru_1/IdentityIdentity!gru_1/Read/ReadVariableOp:value:0*
T0*
_output_shapes

:Z2
gru_1/Identity?
gru_1/Read_1/ReadVariableOpReadVariableOp$gru_1_read_1_readvariableop_resource*
_output_shapes

:Z*
dtype02
gru_1/Read_1/ReadVariableOp~
gru_1/Identity_1Identity#gru_1/Read_1/ReadVariableOp:value:0*
T0*
_output_shapes

:Z2
gru_1/Identity_1?
gru_1/Read_2/ReadVariableOpReadVariableOp$gru_1_read_2_readvariableop_resource*
_output_shapes

:Z*
dtype02
gru_1/Read_2/ReadVariableOp~
gru_1/Identity_2Identity#gru_1/Read_2/ReadVariableOp:value:0*
T0*
_output_shapes

:Z2
gru_1/Identity_2?
gru_1/PartitionedCallPartitionedCallinputsgru_1/zeros:output:0gru_1/Identity:output:0gru_1/Identity_1:output:0gru_1/Identity_2:output:0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *S
_output_shapesA
?:?????????:?????????`:?????????: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *'
f"R 
__inference_standard_gru_842132
gru_1/PartitionedCall?
!dense_10/MLCMatMul/ReadVariableOpReadVariableOp*dense_10_mlcmatmul_readvariableop_resource*
_output_shapes

:0*
dtype02#
!dense_10/MLCMatMul/ReadVariableOp?
dense_10/MLCMatMul	MLCMatMulgru_1/PartitionedCall:output:0)dense_10/MLCMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????02
dense_10/MLCMatMul?
dense_10/BiasAdd/ReadVariableOpReadVariableOp(dense_10_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02!
dense_10/BiasAdd/ReadVariableOp?
dense_10/BiasAddBiasAdddense_10/MLCMatMul:product:0'dense_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????02
dense_10/BiasAdds
dense_10/TanhTanhdense_10/BiasAdd:output:0*
T0*'
_output_shapes
:?????????02
dense_10/Tanh?
IdentityIdentitydense_10/Tanh:y:0 ^dense_10/BiasAdd/ReadVariableOp"^dense_10/MLCMatMul/ReadVariableOp^gru_1/Read/ReadVariableOp^gru_1/Read_1/ReadVariableOp^gru_1/Read_2/ReadVariableOp*
T0*'
_output_shapes
:?????????02

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????`:::::2B
dense_10/BiasAdd/ReadVariableOpdense_10/BiasAdd/ReadVariableOp2F
!dense_10/MLCMatMul/ReadVariableOp!dense_10/MLCMatMul/ReadVariableOp26
gru_1/Read/ReadVariableOpgru_1/Read/ReadVariableOp2:
gru_1/Read_1/ReadVariableOpgru_1/Read_1/ReadVariableOp2:
gru_1/Read_2/ReadVariableOpgru_1/Read_2/ReadVariableOp:S O
+
_output_shapes
:?????????`
 
_user_specified_nameinputs
?M
?
%__forward_gpu_gru_with_fallback_81031

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
T0*H
_output_shapes6
4:`?????????:?????????: :*
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

Identitym

Identity_1Identitytranspose_7_0:y:0*
T0*+
_output_shapes
:?????????`2

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
transpose_permtranspose/perm:output:0*[
_input_shapesJ
H:?????????`:?????????:Z:Z:Z*<
api_implements*(gru_86cc2bcf-3b62-4553-8076-900e106f9134*
api_preferred_deviceGPU*T
backward_function_name:8__inference___backward_gpu_gru_with_fallback_80814_81032*
go_backwards( *

time_major( :S O
+
_output_shapes
:?????????`
 
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
?
G__inference_sequential_8_layer_call_and_return_conditional_losses_83975

inputs
gru_1_83962
gru_1_83964
gru_1_83966
dense_10_83969
dense_10_83971
identity?? dense_10/StatefulPartitionedCall?gru_1/StatefulPartitionedCall?
gru_1/StatefulPartitionedCallStatefulPartitionedCallinputsgru_1_83962gru_1_83964gru_1_83966*
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
GPU 2J 8? *I
fDRB
@__inference_gru_1_layer_call_and_return_conditional_losses_834132
gru_1/StatefulPartitionedCall?
 dense_10/StatefulPartitionedCallStatefulPartitionedCall&gru_1/StatefulPartitionedCall:output:0dense_10_83969dense_10_83971*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????0*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_10_layer_call_and_return_conditional_losses_839232"
 dense_10/StatefulPartitionedCall?
IdentityIdentity)dense_10/StatefulPartitionedCall:output:0!^dense_10/StatefulPartitionedCall^gru_1/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????02

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????`:::::2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2>
gru_1/StatefulPartitionedCallgru_1/StatefulPartitionedCall:S O
+
_output_shapes
:?????????`
 
_user_specified_nameinputs
?2
?
while_body_83491
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
?
G__inference_sequential_8_layer_call_and_return_conditional_losses_83940
gru_1_input
gru_1_83905
gru_1_83907
gru_1_83909
dense_10_83934
dense_10_83936
identity?? dense_10/StatefulPartitionedCall?gru_1/StatefulPartitionedCall?
gru_1/StatefulPartitionedCallStatefulPartitionedCallgru_1_inputgru_1_83905gru_1_83907gru_1_83909*
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
GPU 2J 8? *I
fDRB
@__inference_gru_1_layer_call_and_return_conditional_losses_834132
gru_1/StatefulPartitionedCall?
 dense_10/StatefulPartitionedCallStatefulPartitionedCall&gru_1/StatefulPartitionedCall:output:0dense_10_83934dense_10_83936*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????0*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_10_layer_call_and_return_conditional_losses_839232"
 dense_10/StatefulPartitionedCall?
IdentityIdentity)dense_10/StatefulPartitionedCall:output:0!^dense_10/StatefulPartitionedCall^gru_1/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????02

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????`:::::2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2>
gru_1/StatefulPartitionedCallgru_1/StatefulPartitionedCall:X T
+
_output_shapes
:?????????`
%
_user_specified_namegru_1_input
?
?
@__inference_gru_1_layer_call_and_return_conditional_losses_83413

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
 *S
_output_shapesA
?:?????????:?????????`:?????????: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *'
f"R 
__inference_standard_gru_831132
PartitionedCall?

Identity_3IdentityPartitionedCall:output:0^Read/ReadVariableOp^Read_1/ReadVariableOp^Read_2/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_3"!

identity_3Identity_3:output:0*6
_input_shapes%
#:?????????`:::2*
Read/ReadVariableOpRead/ReadVariableOp2.
Read_1/ReadVariableOpRead_1/ReadVariableOp2.
Read_2/ReadVariableOpRead_2/ReadVariableOp:S O
+
_output_shapes
:?????????`
 
_user_specified_nameinputs
?M
?
%__forward_gpu_gru_with_fallback_83879

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
T0*H
_output_shapes6
4:`?????????:?????????: :*
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

Identitym

Identity_1Identitytranspose_7_0:y:0*
T0*+
_output_shapes
:?????????`2

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
transpose_permtranspose/perm:output:0*[
_input_shapesJ
H:?????????`:?????????:Z:Z:Z*<
api_implements*(gru_4060a475-f519-4bbc-82de-47b15ccf83f2*
api_preferred_deviceGPU*T
backward_function_name:8__inference___backward_gpu_gru_with_fallback_83662_83880*
go_backwards( *

time_major( :S O
+
_output_shapes
:?????????`
 
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
__inference_standard_gru_85664

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
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:`?????????2
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
while_body_85573*
condR
while_cond_85572*R
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
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:`?????????*
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
T0*+
_output_shapes
:?????????`2
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

Identityk

Identity_1Identitytranspose_1:y:0*
T0*+
_output_shapes
:?????????`2

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

identity_3Identity_3:output:0*[
_input_shapesJ
H:?????????`:?????????:Z:Z:Z*<
api_implements*(gru_dfc2bb03-a150-469d-a7e4-08432d7c3f5a*
api_preferred_deviceCPU*
go_backwards( *

time_major( :S O
+
_output_shapes
:?????????`
 
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
while_cond_86532
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice3
/while_while_cond_86532___redundant_placeholder03
/while_while_cond_86532___redundant_placeholder13
/while_while_cond_86532___redundant_placeholder23
/while_while_cond_86532___redundant_placeholder33
/while_while_cond_86532___redundant_placeholder4
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
?M
?
%__forward_gpu_gru_with_fallback_82929

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
api_implements*(gru_696f6c69-2d92-41ae-9f9b-48ad5b13a6ba*
api_preferred_deviceGPU*T
backward_function_name:8__inference___backward_gpu_gru_with_fallback_82712_82930*
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
?
?
%__inference_gru_1_layer_call_fn_85986

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
GPU 2J 8? *I
fDRB
@__inference_gru_1_layer_call_and_return_conditional_losses_838822
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????`:::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????`
 
_user_specified_nameinputs
?<
?
'__inference_gpu_gru_with_fallback_85743

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
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:`?????????2
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
T0*H
_output_shapes6
4:`?????????:?????????: :*
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
T0*+
_output_shapes
:?????????`2
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

Identityk

Identity_1Identitytranspose_7:y:0*
T0*+
_output_shapes
:?????????`2

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

identity_3Identity_3:output:0*[
_input_shapesJ
H:?????????`:?????????:Z:Z:Z*<
api_implements*(gru_dfc2bb03-a150-469d-a7e4-08432d7c3f5a*
api_preferred_deviceGPU*
go_backwards( *

time_major( :S O
+
_output_shapes
:?????????`
 
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
%__forward_gpu_gru_with_fallback_85961

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
T0*H
_output_shapes6
4:`?????????:?????????: :*
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

Identitym

Identity_1Identitytranspose_7_0:y:0*
T0*+
_output_shapes
:?????????`2

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
transpose_permtranspose/perm:output:0*[
_input_shapesJ
H:?????????`:?????????:Z:Z:Z*<
api_implements*(gru_dfc2bb03-a150-469d-a7e4-08432d7c3f5a*
api_preferred_deviceGPU*T
backward_function_name:8__inference___backward_gpu_gru_with_fallback_85744_85962*
go_backwards( *

time_major( :S O
+
_output_shapes
:?????????`
 
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
?'
?
G__inference_sequential_8_layer_call_and_return_conditional_losses_84996

inputs&
"gru_1_read_readvariableop_resource(
$gru_1_read_1_readvariableop_resource(
$gru_1_read_2_readvariableop_resource.
*dense_10_mlcmatmul_readvariableop_resource,
(dense_10_biasadd_readvariableop_resource
identity??dense_10/BiasAdd/ReadVariableOp?!dense_10/MLCMatMul/ReadVariableOp?gru_1/Read/ReadVariableOp?gru_1/Read_1/ReadVariableOp?gru_1/Read_2/ReadVariableOpP
gru_1/ShapeShapeinputs*
T0*
_output_shapes
:2
gru_1/Shape?
gru_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru_1/strided_slice/stack?
gru_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
gru_1/strided_slice/stack_1?
gru_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
gru_1/strided_slice/stack_2?
gru_1/strided_sliceStridedSlicegru_1/Shape:output:0"gru_1/strided_slice/stack:output:0$gru_1/strided_slice/stack_1:output:0$gru_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
gru_1/strided_sliceh
gru_1/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
gru_1/zeros/mul/y?
gru_1/zeros/mulMulgru_1/strided_slice:output:0gru_1/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
gru_1/zeros/mulk
gru_1/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
gru_1/zeros/Less/y
gru_1/zeros/LessLessgru_1/zeros/mul:z:0gru_1/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
gru_1/zeros/Lessn
gru_1/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
gru_1/zeros/packed/1?
gru_1/zeros/packedPackgru_1/strided_slice:output:0gru_1/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
gru_1/zeros/packedk
gru_1/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
gru_1/zeros/Const?
gru_1/zerosFillgru_1/zeros/packed:output:0gru_1/zeros/Const:output:0*
T0*'
_output_shapes
:?????????2
gru_1/zeros?
gru_1/Read/ReadVariableOpReadVariableOp"gru_1_read_readvariableop_resource*
_output_shapes

:Z*
dtype02
gru_1/Read/ReadVariableOpx
gru_1/IdentityIdentity!gru_1/Read/ReadVariableOp:value:0*
T0*
_output_shapes

:Z2
gru_1/Identity?
gru_1/Read_1/ReadVariableOpReadVariableOp$gru_1_read_1_readvariableop_resource*
_output_shapes

:Z*
dtype02
gru_1/Read_1/ReadVariableOp~
gru_1/Identity_1Identity#gru_1/Read_1/ReadVariableOp:value:0*
T0*
_output_shapes

:Z2
gru_1/Identity_1?
gru_1/Read_2/ReadVariableOpReadVariableOp$gru_1_read_2_readvariableop_resource*
_output_shapes

:Z*
dtype02
gru_1/Read_2/ReadVariableOp~
gru_1/Identity_2Identity#gru_1/Read_2/ReadVariableOp:value:0*
T0*
_output_shapes

:Z2
gru_1/Identity_2?
gru_1/PartitionedCallPartitionedCallinputsgru_1/zeros:output:0gru_1/Identity:output:0gru_1/Identity_1:output:0gru_1/Identity_2:output:0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *S
_output_shapesA
?:?????????:?????????`:?????????: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *'
f"R 
__inference_standard_gru_846892
gru_1/PartitionedCall?
!dense_10/MLCMatMul/ReadVariableOpReadVariableOp*dense_10_mlcmatmul_readvariableop_resource*
_output_shapes

:0*
dtype02#
!dense_10/MLCMatMul/ReadVariableOp?
dense_10/MLCMatMul	MLCMatMulgru_1/PartitionedCall:output:0)dense_10/MLCMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????02
dense_10/MLCMatMul?
dense_10/BiasAdd/ReadVariableOpReadVariableOp(dense_10_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02!
dense_10/BiasAdd/ReadVariableOp?
dense_10/BiasAddBiasAdddense_10/MLCMatMul:product:0'dense_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????02
dense_10/BiasAdds
dense_10/TanhTanhdense_10/BiasAdd:output:0*
T0*'
_output_shapes
:?????????02
dense_10/Tanh?
IdentityIdentitydense_10/Tanh:y:0 ^dense_10/BiasAdd/ReadVariableOp"^dense_10/MLCMatMul/ReadVariableOp^gru_1/Read/ReadVariableOp^gru_1/Read_1/ReadVariableOp^gru_1/Read_2/ReadVariableOp*
T0*'
_output_shapes
:?????????02

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????`:::::2B
dense_10/BiasAdd/ReadVariableOpdense_10/BiasAdd/ReadVariableOp2F
!dense_10/MLCMatMul/ReadVariableOp!dense_10/MLCMatMul/ReadVariableOp26
gru_1/Read/ReadVariableOpgru_1/Read/ReadVariableOp2:
gru_1/Read_1/ReadVariableOpgru_1/Read_1/ReadVariableOp2:
gru_1/Read_2/ReadVariableOpgru_1/Read_2/ReadVariableOp:S O
+
_output_shapes
:?????????`
 
_user_specified_nameinputs
?
?
@__inference_gru_1_layer_call_and_return_conditional_losses_85495

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
 *S
_output_shapesA
?:?????????:?????????`:?????????: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *'
f"R 
__inference_standard_gru_851952
PartitionedCall?

Identity_3IdentityPartitionedCall:output:0^Read/ReadVariableOp^Read_1/ReadVariableOp^Read_2/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_3"!

identity_3Identity_3:output:0*6
_input_shapes%
#:?????????`:::2*
Read/ReadVariableOpRead/ReadVariableOp2.
Read_1/ReadVariableOpRead_1/ReadVariableOp2.
Read_2/ReadVariableOpRead_2/ReadVariableOp:S O
+
_output_shapes
:?????????`
 
_user_specified_nameinputs
ҭ
?
8__inference___backward_gpu_gru_with_fallback_84293_84511
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
gradients/grad_ys_0{
gradients/grad_ys_1Identityplaceholder_1*
T0*+
_output_shapes
:?????????`2
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
T0*+
_output_shapes
:`?????????*
shrink_axis_mask22
0gradients/strided_slice_grad/MLCStridedSliceGrad?
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:2.
,gradients/transpose_7_grad/InvertPermutation?
+gradients/transpose_7_grad/MLCTransposeGradMLCTransposeGradgradients/grad_ys_1:output:00gradients/transpose_7_grad/InvertPermutation:y:07gradients_transpose_7_grad_mlctransposegrad_transpose_7*
T0*+
_output_shapes
:`?????????2-
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
75loc:@gradients/strided_slice_grad/MLCStridedSliceGrad*+
_output_shapes
:`?????????2
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
T0*K
_output_shapes9
7:`?????????:?????????: :?*
rnn_modegru2*
(gradients/CudnnRNN_grad/CudnnRNNBackprop?
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:2,
*gradients/transpose_grad/InvertPermutation?
)gradients/transpose_grad/MLCTransposeGradMLCTransposeGrad9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:02gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose*
T0*+
_output_shapes
:?????????`2+
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
T0*+
_output_shapes
:?????????`2

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
?:?????????:?????????`:?????????: :`?????????:?????????::?????????`:?????????: ::`?????????:?????????: :?::?????????: :?:?:?:?:?:?::::::::::::::::::: : :: ::?*<
api_implements*(gru_b643a1c6-e6ee-4f24-b167-0aaa19385eef*
api_preferred_deviceGPU*@
forward_function_name'%__forward_gpu_gru_with_fallback_84510*
go_backwards( *

time_major( :- )
'
_output_shapes
:?????????:1-
+
_output_shapes
:?????????`:-)
'
_output_shapes
:?????????:

_output_shapes
: :1-
+
_output_shapes
:`?????????:-)
'
_output_shapes
:?????????: 

_output_shapes
::1-
+
_output_shapes
:?????????`:1-
+
_output_shapes
:?????????:	

_output_shapes
: :


_output_shapes
::1-
+
_output_shapes
:`?????????:1-
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
,__inference_sequential_8_layer_call_fn_85011

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
 *'
_output_shapes
:?????????0*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_sequential_8_layer_call_and_return_conditional_losses_839752
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????02

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????`:::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????`
 
_user_specified_nameinputs
?
?
@__inference_gru_1_layer_call_and_return_conditional_losses_82932

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
__inference_standard_gru_826322
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
?<
?
'__inference_gpu_gru_with_fallback_80813

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
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:`?????????2
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
T0*H
_output_shapes6
4:`?????????:?????????: :*
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
T0*+
_output_shapes
:?????????`2
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

Identityk

Identity_1Identitytranspose_7:y:0*
T0*+
_output_shapes
:?????????`2

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

identity_3Identity_3:output:0*[
_input_shapesJ
H:?????????`:?????????:Z:Z:Z*<
api_implements*(gru_86cc2bcf-3b62-4553-8076-900e106f9134*
api_preferred_deviceGPU*
go_backwards( *

time_major( :S O
+
_output_shapes
:?????????`
 
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
while_cond_86063
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice3
/while_while_cond_86063___redundant_placeholder03
/while_while_cond_86063___redundant_placeholder13
/while_while_cond_86063___redundant_placeholder23
/while_while_cond_86063___redundant_placeholder33
/while_while_cond_86063___redundant_placeholder4
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
ҭ
?
8__inference___backward_gpu_gru_with_fallback_83193_83411
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
gradients/grad_ys_0{
gradients/grad_ys_1Identityplaceholder_1*
T0*+
_output_shapes
:?????????`2
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
T0*+
_output_shapes
:`?????????*
shrink_axis_mask22
0gradients/strided_slice_grad/MLCStridedSliceGrad?
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:2.
,gradients/transpose_7_grad/InvertPermutation?
+gradients/transpose_7_grad/MLCTransposeGradMLCTransposeGradgradients/grad_ys_1:output:00gradients/transpose_7_grad/InvertPermutation:y:07gradients_transpose_7_grad_mlctransposegrad_transpose_7*
T0*+
_output_shapes
:`?????????2-
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
75loc:@gradients/strided_slice_grad/MLCStridedSliceGrad*+
_output_shapes
:`?????????2
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
T0*K
_output_shapes9
7:`?????????:?????????: :?*
rnn_modegru2*
(gradients/CudnnRNN_grad/CudnnRNNBackprop?
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:2,
*gradients/transpose_grad/InvertPermutation?
)gradients/transpose_grad/MLCTransposeGradMLCTransposeGrad9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:02gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose*
T0*+
_output_shapes
:?????????`2+
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
T0*+
_output_shapes
:?????????`2

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
?:?????????:?????????`:?????????: :`?????????:?????????::?????????`:?????????: ::`?????????:?????????: :?::?????????: :?:?:?:?:?:?::::::::::::::::::: : :: ::?*<
api_implements*(gru_468cd0a2-ab9a-4860-82a2-a36e174c1ce4*
api_preferred_deviceGPU*@
forward_function_name'%__forward_gpu_gru_with_fallback_83410*
go_backwards( *

time_major( :- )
'
_output_shapes
:?????????:1-
+
_output_shapes
:?????????`:-)
'
_output_shapes
:?????????:

_output_shapes
: :1-
+
_output_shapes
:`?????????:-)
'
_output_shapes
:?????????: 

_output_shapes
::1-
+
_output_shapes
:?????????`:1-
+
_output_shapes
:?????????:	

_output_shapes
: :


_output_shapes
::1-
+
_output_shapes
:`?????????:1-
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
while_body_86064
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
?
}
(__inference_dense_10_layer_call_fn_86966

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
 *'
_output_shapes
:?????????0*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_10_layer_call_and_return_conditional_losses_839232
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????02

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
,__inference_sequential_8_layer_call_fn_84019
gru_1_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallgru_1_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????0*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_sequential_8_layer_call_and_return_conditional_losses_840062
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????02

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????`:::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
+
_output_shapes
:?????????`
%
_user_specified_namegru_1_input
ҭ
?
8__inference___backward_gpu_gru_with_fallback_84769_84987
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
gradients/grad_ys_0{
gradients/grad_ys_1Identityplaceholder_1*
T0*+
_output_shapes
:?????????`2
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
T0*+
_output_shapes
:`?????????*
shrink_axis_mask22
0gradients/strided_slice_grad/MLCStridedSliceGrad?
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:2.
,gradients/transpose_7_grad/InvertPermutation?
+gradients/transpose_7_grad/MLCTransposeGradMLCTransposeGradgradients/grad_ys_1:output:00gradients/transpose_7_grad/InvertPermutation:y:07gradients_transpose_7_grad_mlctransposegrad_transpose_7*
T0*+
_output_shapes
:`?????????2-
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
75loc:@gradients/strided_slice_grad/MLCStridedSliceGrad*+
_output_shapes
:`?????????2
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
T0*K
_output_shapes9
7:`?????????:?????????: :?*
rnn_modegru2*
(gradients/CudnnRNN_grad/CudnnRNNBackprop?
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:2,
*gradients/transpose_grad/InvertPermutation?
)gradients/transpose_grad/MLCTransposeGradMLCTransposeGrad9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:02gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose*
T0*+
_output_shapes
:?????????`2+
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
T0*+
_output_shapes
:?????????`2

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
?:?????????:?????????`:?????????: :`?????????:?????????::?????????`:?????????: ::`?????????:?????????: :?::?????????: :?:?:?:?:?:?::::::::::::::::::: : :: ::?*<
api_implements*(gru_3c83e14b-7d3f-480b-af56-3855cc00958f*
api_preferred_deviceGPU*@
forward_function_name'%__forward_gpu_gru_with_fallback_84986*
go_backwards( *

time_major( :- )
'
_output_shapes
:?????????:1-
+
_output_shapes
:?????????`:-)
'
_output_shapes
:?????????:

_output_shapes
: :1-
+
_output_shapes
:`?????????:-)
'
_output_shapes
:?????????: 

_output_shapes
::1-
+
_output_shapes
:?????????`:1-
+
_output_shapes
:?????????:	

_output_shapes
: :


_output_shapes
::1-
+
_output_shapes
:`?????????:1-
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
?=
?
'__inference_gpu_gru_with_fallback_82711

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
api_implements*(gru_696f6c69-2d92-41ae-9f9b-48ad5b13a6ba*
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
?
?
G__inference_sequential_8_layer_call_and_return_conditional_losses_83956
gru_1_input
gru_1_83943
gru_1_83945
gru_1_83947
dense_10_83950
dense_10_83952
identity?? dense_10/StatefulPartitionedCall?gru_1/StatefulPartitionedCall?
gru_1/StatefulPartitionedCallStatefulPartitionedCallgru_1_inputgru_1_83943gru_1_83945gru_1_83947*
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
GPU 2J 8? *I
fDRB
@__inference_gru_1_layer_call_and_return_conditional_losses_838822
gru_1/StatefulPartitionedCall?
 dense_10/StatefulPartitionedCallStatefulPartitionedCall&gru_1/StatefulPartitionedCall:output:0dense_10_83950dense_10_83952*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????0*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_10_layer_call_and_return_conditional_losses_839232"
 dense_10/StatefulPartitionedCall?
IdentityIdentity)dense_10/StatefulPartitionedCall:output:0!^dense_10/StatefulPartitionedCall^gru_1/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????02

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????`:::::2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2>
gru_1/StatefulPartitionedCallgru_1/StatefulPartitionedCall:X T
+
_output_shapes
:?????????`
%
_user_specified_namegru_1_input
?E
?
__inference_standard_gru_83113

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
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:`?????????2
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
while_body_83022*
condR
while_cond_83021*R
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
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:`?????????*
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
T0*+
_output_shapes
:?????????`2
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

Identityk

Identity_1Identitytranspose_1:y:0*
T0*+
_output_shapes
:?????????`2

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

identity_3Identity_3:output:0*[
_input_shapesJ
H:?????????`:?????????:Z:Z:Z*<
api_implements*(gru_468cd0a2-ab9a-4860-82a2-a36e174c1ce4*
api_preferred_deviceCPU*
go_backwards( *

time_major( :S O
+
_output_shapes
:?????????`
 
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
__inference_standard_gru_84213

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
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:`?????????2
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
while_body_84122*
condR
while_cond_84121*R
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
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:`?????????*
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
T0*+
_output_shapes
:?????????`2
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

Identityk

Identity_1Identitytranspose_1:y:0*
T0*+
_output_shapes
:?????????`2

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

identity_3Identity_3:output:0*[
_input_shapesJ
H:?????????`:?????????:Z:Z:Z*<
api_implements*(gru_b643a1c6-e6ee-4f24-b167-0aaa19385eef*
api_preferred_deviceCPU*
go_backwards( *

time_major( :S O
+
_output_shapes
:?????????`
 
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
ҭ
?
8__inference___backward_gpu_gru_with_fallback_83662_83880
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
gradients/grad_ys_0{
gradients/grad_ys_1Identityplaceholder_1*
T0*+
_output_shapes
:?????????`2
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
T0*+
_output_shapes
:`?????????*
shrink_axis_mask22
0gradients/strided_slice_grad/MLCStridedSliceGrad?
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:2.
,gradients/transpose_7_grad/InvertPermutation?
+gradients/transpose_7_grad/MLCTransposeGradMLCTransposeGradgradients/grad_ys_1:output:00gradients/transpose_7_grad/InvertPermutation:y:07gradients_transpose_7_grad_mlctransposegrad_transpose_7*
T0*+
_output_shapes
:`?????????2-
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
75loc:@gradients/strided_slice_grad/MLCStridedSliceGrad*+
_output_shapes
:`?????????2
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
T0*K
_output_shapes9
7:`?????????:?????????: :?*
rnn_modegru2*
(gradients/CudnnRNN_grad/CudnnRNNBackprop?
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:2,
*gradients/transpose_grad/InvertPermutation?
)gradients/transpose_grad/MLCTransposeGradMLCTransposeGrad9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:02gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose*
T0*+
_output_shapes
:?????????`2+
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
T0*+
_output_shapes
:?????????`2

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
?:?????????:?????????`:?????????: :`?????????:?????????::?????????`:?????????: ::`?????????:?????????: :?::?????????: :?:?:?:?:?:?::::::::::::::::::: : :: ::?*<
api_implements*(gru_4060a475-f519-4bbc-82de-47b15ccf83f2*
api_preferred_deviceGPU*@
forward_function_name'%__forward_gpu_gru_with_fallback_83879*
go_backwards( *

time_major( :- )
'
_output_shapes
:?????????:1-
+
_output_shapes
:?????????`:-)
'
_output_shapes
:?????????:

_output_shapes
: :1-
+
_output_shapes
:`?????????:-)
'
_output_shapes
:?????????: 

_output_shapes
::1-
+
_output_shapes
:?????????`:1-
+
_output_shapes
:?????????:	

_output_shapes
: :


_output_shapes
::1-
+
_output_shapes
:`?????????:1-
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
?
?
@__inference_gru_1_layer_call_and_return_conditional_losses_85964

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
 *S
_output_shapesA
?:?????????:?????????`:?????????: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *'
f"R 
__inference_standard_gru_856642
PartitionedCall?

Identity_3IdentityPartitionedCall:output:0^Read/ReadVariableOp^Read_1/ReadVariableOp^Read_2/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_3"!

identity_3Identity_3:output:0*6
_input_shapes%
#:?????????`:::2*
Read/ReadVariableOpRead/ReadVariableOp2.
Read_1/ReadVariableOpRead_1/ReadVariableOp2.
Read_2/ReadVariableOpRead_2/ReadVariableOp:S O
+
_output_shapes
:?????????`
 
_user_specified_nameinputs
?
?
@__inference_gru_1_layer_call_and_return_conditional_losses_86455
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
__inference_standard_gru_861552
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
?2
?
 __inference__wrapped_model_81041
gru_1_input3
/sequential_8_gru_1_read_readvariableop_resource5
1sequential_8_gru_1_read_1_readvariableop_resource5
1sequential_8_gru_1_read_2_readvariableop_resource;
7sequential_8_dense_10_mlcmatmul_readvariableop_resource9
5sequential_8_dense_10_biasadd_readvariableop_resource
identity??,sequential_8/dense_10/BiasAdd/ReadVariableOp?.sequential_8/dense_10/MLCMatMul/ReadVariableOp?&sequential_8/gru_1/Read/ReadVariableOp?(sequential_8/gru_1/Read_1/ReadVariableOp?(sequential_8/gru_1/Read_2/ReadVariableOpo
sequential_8/gru_1/ShapeShapegru_1_input*
T0*
_output_shapes
:2
sequential_8/gru_1/Shape?
&sequential_8/gru_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential_8/gru_1/strided_slice/stack?
(sequential_8/gru_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(sequential_8/gru_1/strided_slice/stack_1?
(sequential_8/gru_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(sequential_8/gru_1/strided_slice/stack_2?
 sequential_8/gru_1/strided_sliceStridedSlice!sequential_8/gru_1/Shape:output:0/sequential_8/gru_1/strided_slice/stack:output:01sequential_8/gru_1/strided_slice/stack_1:output:01sequential_8/gru_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 sequential_8/gru_1/strided_slice?
sequential_8/gru_1/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2 
sequential_8/gru_1/zeros/mul/y?
sequential_8/gru_1/zeros/mulMul)sequential_8/gru_1/strided_slice:output:0'sequential_8/gru_1/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
sequential_8/gru_1/zeros/mul?
sequential_8/gru_1/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2!
sequential_8/gru_1/zeros/Less/y?
sequential_8/gru_1/zeros/LessLess sequential_8/gru_1/zeros/mul:z:0(sequential_8/gru_1/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
sequential_8/gru_1/zeros/Less?
!sequential_8/gru_1/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2#
!sequential_8/gru_1/zeros/packed/1?
sequential_8/gru_1/zeros/packedPack)sequential_8/gru_1/strided_slice:output:0*sequential_8/gru_1/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2!
sequential_8/gru_1/zeros/packed?
sequential_8/gru_1/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
sequential_8/gru_1/zeros/Const?
sequential_8/gru_1/zerosFill(sequential_8/gru_1/zeros/packed:output:0'sequential_8/gru_1/zeros/Const:output:0*
T0*'
_output_shapes
:?????????2
sequential_8/gru_1/zeros?
&sequential_8/gru_1/Read/ReadVariableOpReadVariableOp/sequential_8_gru_1_read_readvariableop_resource*
_output_shapes

:Z*
dtype02(
&sequential_8/gru_1/Read/ReadVariableOp?
sequential_8/gru_1/IdentityIdentity.sequential_8/gru_1/Read/ReadVariableOp:value:0*
T0*
_output_shapes

:Z2
sequential_8/gru_1/Identity?
(sequential_8/gru_1/Read_1/ReadVariableOpReadVariableOp1sequential_8_gru_1_read_1_readvariableop_resource*
_output_shapes

:Z*
dtype02*
(sequential_8/gru_1/Read_1/ReadVariableOp?
sequential_8/gru_1/Identity_1Identity0sequential_8/gru_1/Read_1/ReadVariableOp:value:0*
T0*
_output_shapes

:Z2
sequential_8/gru_1/Identity_1?
(sequential_8/gru_1/Read_2/ReadVariableOpReadVariableOp1sequential_8_gru_1_read_2_readvariableop_resource*
_output_shapes

:Z*
dtype02*
(sequential_8/gru_1/Read_2/ReadVariableOp?
sequential_8/gru_1/Identity_2Identity0sequential_8/gru_1/Read_2/ReadVariableOp:value:0*
T0*
_output_shapes

:Z2
sequential_8/gru_1/Identity_2?
"sequential_8/gru_1/PartitionedCallPartitionedCallgru_1_input!sequential_8/gru_1/zeros:output:0$sequential_8/gru_1/Identity:output:0&sequential_8/gru_1/Identity_1:output:0&sequential_8/gru_1/Identity_2:output:0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *S
_output_shapesA
?:?????????:?????????`:?????????: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *'
f"R 
__inference_standard_gru_807342$
"sequential_8/gru_1/PartitionedCall?
.sequential_8/dense_10/MLCMatMul/ReadVariableOpReadVariableOp7sequential_8_dense_10_mlcmatmul_readvariableop_resource*
_output_shapes

:0*
dtype020
.sequential_8/dense_10/MLCMatMul/ReadVariableOp?
sequential_8/dense_10/MLCMatMul	MLCMatMul+sequential_8/gru_1/PartitionedCall:output:06sequential_8/dense_10/MLCMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????02!
sequential_8/dense_10/MLCMatMul?
,sequential_8/dense_10/BiasAdd/ReadVariableOpReadVariableOp5sequential_8_dense_10_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02.
,sequential_8/dense_10/BiasAdd/ReadVariableOp?
sequential_8/dense_10/BiasAddBiasAdd)sequential_8/dense_10/MLCMatMul:product:04sequential_8/dense_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????02
sequential_8/dense_10/BiasAdd?
sequential_8/dense_10/TanhTanh&sequential_8/dense_10/BiasAdd:output:0*
T0*'
_output_shapes
:?????????02
sequential_8/dense_10/Tanh?
IdentityIdentitysequential_8/dense_10/Tanh:y:0-^sequential_8/dense_10/BiasAdd/ReadVariableOp/^sequential_8/dense_10/MLCMatMul/ReadVariableOp'^sequential_8/gru_1/Read/ReadVariableOp)^sequential_8/gru_1/Read_1/ReadVariableOp)^sequential_8/gru_1/Read_2/ReadVariableOp*
T0*'
_output_shapes
:?????????02

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????`:::::2\
,sequential_8/dense_10/BiasAdd/ReadVariableOp,sequential_8/dense_10/BiasAdd/ReadVariableOp2`
.sequential_8/dense_10/MLCMatMul/ReadVariableOp.sequential_8/dense_10/MLCMatMul/ReadVariableOp2P
&sequential_8/gru_1/Read/ReadVariableOp&sequential_8/gru_1/Read/ReadVariableOp2T
(sequential_8/gru_1/Read_1/ReadVariableOp(sequential_8/gru_1/Read_1/ReadVariableOp2T
(sequential_8/gru_1/Read_2/ReadVariableOp(sequential_8/gru_1/Read_2/ReadVariableOp:X T
+
_output_shapes
:?????????`
%
_user_specified_namegru_1_input
?2
?
while_body_83022
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
?E
?
__inference_standard_gru_84689

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
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:`?????????2
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
while_body_84598*
condR
while_cond_84597*R
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
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:`?????????*
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
T0*+
_output_shapes
:?????????`2
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

Identityk

Identity_1Identitytranspose_1:y:0*
T0*+
_output_shapes
:?????????`2

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

identity_3Identity_3:output:0*[
_input_shapesJ
H:?????????`:?????????:Z:Z:Z*<
api_implements*(gru_3c83e14b-7d3f-480b-af56-3855cc00958f*
api_preferred_deviceCPU*
go_backwards( *

time_major( :S O
+
_output_shapes
:?????????`
 
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
C__inference_dense_10_layer_call_and_return_conditional_losses_83923

inputs%
!mlcmatmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MLCMatMul/ReadVariableOp?
MLCMatMul/ReadVariableOpReadVariableOp!mlcmatmul_readvariableop_resource*
_output_shapes

:0*
dtype02
MLCMatMul/ReadVariableOp
	MLCMatMul	MLCMatMulinputs MLCMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????02
	MLCMatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:0*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMLCMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????02	
BiasAddX
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:?????????02
Tanh?
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MLCMatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????02

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
?E
?
__inference_standard_gru_85195

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
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:`?????????2
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
while_body_85104*
condR
while_cond_85103*R
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
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:`?????????*
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
T0*+
_output_shapes
:?????????`2
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

Identityk

Identity_1Identitytranspose_1:y:0*
T0*+
_output_shapes
:?????????`2

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

identity_3Identity_3:output:0*[
_input_shapesJ
H:?????????`:?????????:Z:Z:Z*<
api_implements*(gru_8967532f-e4ff-47ed-bad4-d5573368c1bb*
api_preferred_deviceCPU*
go_backwards( *

time_major( :S O
+
_output_shapes
:?????????`
 
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
@__inference_gru_1_layer_call_and_return_conditional_losses_86924
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
__inference_standard_gru_866242
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
?E
?
__inference_standard_gru_83582

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
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:`?????????2
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
while_body_83491*
condR
while_cond_83490*R
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
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:`?????????*
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
T0*+
_output_shapes
:?????????`2
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

Identityk

Identity_1Identitytranspose_1:y:0*
T0*+
_output_shapes
:?????????`2

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

identity_3Identity_3:output:0*[
_input_shapesJ
H:?????????`:?????????:Z:Z:Z*<
api_implements*(gru_4060a475-f519-4bbc-82de-47b15ccf83f2*
api_preferred_deviceCPU*
go_backwards( *

time_major( :S O
+
_output_shapes
:?????????`
 
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
while_cond_82540
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice3
/while_while_cond_82540___redundant_placeholder03
/while_while_cond_82540___redundant_placeholder13
/while_while_cond_82540___redundant_placeholder23
/while_while_cond_82540___redundant_placeholder33
/while_while_cond_82540___redundant_placeholder4
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
?=
?
'__inference_gpu_gru_with_fallback_86234

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
api_implements*(gru_dbdfb1bc-8350-49bc-99d4-4927254488fc*
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
%__forward_gpu_gru_with_fallback_85492

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
T0*H
_output_shapes6
4:`?????????:?????????: :*
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

Identitym

Identity_1Identitytranspose_7_0:y:0*
T0*+
_output_shapes
:?????????`2

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
transpose_permtranspose/perm:output:0*[
_input_shapesJ
H:?????????`:?????????:Z:Z:Z*<
api_implements*(gru_8967532f-e4ff-47ed-bad4-d5573368c1bb*
api_preferred_deviceGPU*T
backward_function_name:8__inference___backward_gpu_gru_with_fallback_85275_85493*
go_backwards( *

time_major( :S O
+
_output_shapes
:?????????`
 
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
%__forward_gpu_gru_with_fallback_83410

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
T0*H
_output_shapes6
4:`?????????:?????????: :*
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

Identitym

Identity_1Identitytranspose_7_0:y:0*
T0*+
_output_shapes
:?????????`2

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
transpose_permtranspose/perm:output:0*[
_input_shapesJ
H:?????????`:?????????:Z:Z:Z*<
api_implements*(gru_468cd0a2-ab9a-4860-82a2-a36e174c1ce4*
api_preferred_deviceGPU*T
backward_function_name:8__inference___backward_gpu_gru_with_fallback_83193_83411*
go_backwards( *

time_major( :S O
+
_output_shapes
:?????????`
 
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
while_body_85104
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
'__inference_gpu_gru_with_fallback_83192

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
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:`?????????2
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
T0*H
_output_shapes6
4:`?????????:?????????: :*
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
T0*+
_output_shapes
:?????????`2
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

Identityk

Identity_1Identitytranspose_7:y:0*
T0*+
_output_shapes
:?????????`2

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

identity_3Identity_3:output:0*[
_input_shapesJ
H:?????????`:?????????:Z:Z:Z*<
api_implements*(gru_468cd0a2-ab9a-4860-82a2-a36e174c1ce4*
api_preferred_deviceGPU*
go_backwards( *

time_major( :S O
+
_output_shapes
:?????????`
 
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
@__inference_gru_1_layer_call_and_return_conditional_losses_83882

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
 *S
_output_shapesA
?:?????????:?????????`:?????????: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *'
f"R 
__inference_standard_gru_835822
PartitionedCall?

Identity_3IdentityPartitionedCall:output:0^Read/ReadVariableOp^Read_1/ReadVariableOp^Read_2/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_3"!

identity_3Identity_3:output:0*6
_input_shapes%
#:?????????`:::2*
Read/ReadVariableOpRead/ReadVariableOp2.
Read_1/ReadVariableOpRead_1/ReadVariableOp2.
Read_2/ReadVariableOpRead_2/ReadVariableOp:S O
+
_output_shapes
:?????????`
 
_user_specified_nameinputs
?<
?
'__inference_gpu_gru_with_fallback_85274

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
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:`?????????2
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
T0*H
_output_shapes6
4:`?????????:?????????: :*
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
T0*+
_output_shapes
:?????????`2
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

Identityk

Identity_1Identitytranspose_7:y:0*
T0*+
_output_shapes
:?????????`2

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

identity_3Identity_3:output:0*[
_input_shapesJ
H:?????????`:?????????:Z:Z:Z*<
api_implements*(gru_8967532f-e4ff-47ed-bad4-d5573368c1bb*
api_preferred_deviceGPU*
go_backwards( *

time_major( :S O
+
_output_shapes
:?????????`
 
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
ҭ
?
8__inference___backward_gpu_gru_with_fallback_85275_85493
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
gradients/grad_ys_0{
gradients/grad_ys_1Identityplaceholder_1*
T0*+
_output_shapes
:?????????`2
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
T0*+
_output_shapes
:`?????????*
shrink_axis_mask22
0gradients/strided_slice_grad/MLCStridedSliceGrad?
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:2.
,gradients/transpose_7_grad/InvertPermutation?
+gradients/transpose_7_grad/MLCTransposeGradMLCTransposeGradgradients/grad_ys_1:output:00gradients/transpose_7_grad/InvertPermutation:y:07gradients_transpose_7_grad_mlctransposegrad_transpose_7*
T0*+
_output_shapes
:`?????????2-
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
75loc:@gradients/strided_slice_grad/MLCStridedSliceGrad*+
_output_shapes
:`?????????2
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
T0*K
_output_shapes9
7:`?????????:?????????: :?*
rnn_modegru2*
(gradients/CudnnRNN_grad/CudnnRNNBackprop?
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:2,
*gradients/transpose_grad/InvertPermutation?
)gradients/transpose_grad/MLCTransposeGradMLCTransposeGrad9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:02gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose*
T0*+
_output_shapes
:?????????`2+
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
T0*+
_output_shapes
:?????????`2

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
?:?????????:?????????`:?????????: :`?????????:?????????::?????????`:?????????: ::`?????????:?????????: :?::?????????: :?:?:?:?:?:?::::::::::::::::::: : :: ::?*<
api_implements*(gru_8967532f-e4ff-47ed-bad4-d5573368c1bb*
api_preferred_deviceGPU*@
forward_function_name'%__forward_gpu_gru_with_fallback_85492*
go_backwards( *

time_major( :- )
'
_output_shapes
:?????????:1-
+
_output_shapes
:?????????`:-)
'
_output_shapes
:?????????:

_output_shapes
: :1-
+
_output_shapes
:`?????????:-)
'
_output_shapes
:?????????: 

_output_shapes
::1-
+
_output_shapes
:?????????`:1-
+
_output_shapes
:?????????:	

_output_shapes
: :


_output_shapes
::1-
+
_output_shapes
:`?????????:1-
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
%__forward_gpu_gru_with_fallback_84986

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
T0*H
_output_shapes6
4:`?????????:?????????: :*
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

Identitym

Identity_1Identitytranspose_7_0:y:0*
T0*+
_output_shapes
:?????????`2

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
transpose_permtranspose/perm:output:0*[
_input_shapesJ
H:?????????`:?????????:Z:Z:Z*<
api_implements*(gru_3c83e14b-7d3f-480b-af56-3855cc00958f*
api_preferred_deviceGPU*T
backward_function_name:8__inference___backward_gpu_gru_with_fallback_84769_84987*
go_backwards( *

time_major( :S O
+
_output_shapes
:?????????`
 
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
__inference_standard_gru_80734

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
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:`?????????2
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
while_body_80643*
condR
while_cond_80642*R
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
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:`?????????*
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
T0*+
_output_shapes
:?????????`2
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

Identityk

Identity_1Identitytranspose_1:y:0*
T0*+
_output_shapes
:?????????`2

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

identity_3Identity_3:output:0*[
_input_shapesJ
H:?????????`:?????????:Z:Z:Z*<
api_implements*(gru_86cc2bcf-3b62-4553-8076-900e106f9134*
api_preferred_deviceCPU*
go_backwards( *

time_major( :S O
+
_output_shapes
:?????????`
 
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
?n
?
!__inference__traced_restore_87155
file_prefix$
 assignvariableop_dense_10_kernel$
 assignvariableop_1_dense_10_bias 
assignvariableop_2_adam_iter"
assignvariableop_3_adam_beta_1"
assignvariableop_4_adam_beta_2!
assignvariableop_5_adam_decay)
%assignvariableop_6_adam_learning_rate.
*assignvariableop_7_gru_1_gru_cell_2_kernel8
4assignvariableop_8_gru_1_gru_cell_2_recurrent_kernel,
(assignvariableop_9_gru_1_gru_cell_2_bias
assignvariableop_10_total
assignvariableop_11_count
assignvariableop_12_total_1
assignvariableop_13_count_1
assignvariableop_14_total_2
assignvariableop_15_count_2.
*assignvariableop_16_adam_dense_10_kernel_m,
(assignvariableop_17_adam_dense_10_bias_m6
2assignvariableop_18_adam_gru_1_gru_cell_2_kernel_m@
<assignvariableop_19_adam_gru_1_gru_cell_2_recurrent_kernel_m4
0assignvariableop_20_adam_gru_1_gru_cell_2_bias_m.
*assignvariableop_21_adam_dense_10_kernel_v,
(assignvariableop_22_adam_dense_10_bias_v6
2assignvariableop_23_adam_gru_1_gru_cell_2_kernel_v@
<assignvariableop_24_adam_gru_1_gru_cell_2_recurrent_kernel_v4
0assignvariableop_25_adam_gru_1_gru_cell_2_bias_v
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
AssignVariableOpAssignVariableOp assignvariableop_dense_10_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_10_biasIdentity_1:output:0"/device:CPU:0*
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
AssignVariableOp_7AssignVariableOp*assignvariableop_7_gru_1_gru_cell_2_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp4assignvariableop_8_gru_1_gru_cell_2_recurrent_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp(assignvariableop_9_gru_1_gru_cell_2_biasIdentity_9:output:0"/device:CPU:0*
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
AssignVariableOp_16AssignVariableOp*assignvariableop_16_adam_dense_10_kernel_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp(assignvariableop_17_adam_dense_10_bias_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp2assignvariableop_18_adam_gru_1_gru_cell_2_kernel_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp<assignvariableop_19_adam_gru_1_gru_cell_2_recurrent_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp0assignvariableop_20_adam_gru_1_gru_cell_2_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp*assignvariableop_21_adam_dense_10_kernel_vIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp(assignvariableop_22_adam_dense_10_bias_vIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp2assignvariableop_23_adam_gru_1_gru_cell_2_kernel_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp<assignvariableop_24_adam_gru_1_gru_cell_2_recurrent_kernel_vIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp0assignvariableop_25_adam_gru_1_gru_cell_2_bias_vIdentity_25:output:0"/device:CPU:0*
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
?2
?
while_body_84122
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
while_cond_85572
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice3
/while_while_cond_85572___redundant_placeholder03
/while_while_cond_85572___redundant_placeholder13
/while_while_cond_85572___redundant_placeholder23
/while_while_cond_85572___redundant_placeholder33
/while_while_cond_85572___redundant_placeholder4
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
@__inference_gru_1_layer_call_and_return_conditional_losses_82452

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
__inference_standard_gru_821522
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
?	
?
while_cond_83490
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice3
/while_while_cond_83490___redundant_placeholder03
/while_while_cond_83490___redundant_placeholder13
/while_while_cond_83490___redundant_placeholder23
/while_while_cond_83490___redundant_placeholder33
/while_while_cond_83490___redundant_placeholder4
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
8__inference___backward_gpu_gru_with_fallback_86704_86922
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
api_implements*(gru_8264b2e4-d807-4ce4-88e8-68c6457615e8*
api_preferred_deviceGPU*@
forward_function_name'%__forward_gpu_gru_with_fallback_86921*
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
ڮ
?
8__inference___backward_gpu_gru_with_fallback_82712_82930
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
api_implements*(gru_696f6c69-2d92-41ae-9f9b-48ad5b13a6ba*
api_preferred_deviceGPU*@
forward_function_name'%__forward_gpu_gru_with_fallback_82929*
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
#__inference_signature_wrapper_84044
gru_1_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallgru_1_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????0*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8? *)
f$R"
 __inference__wrapped_model_810412
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????02

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????`:::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
+
_output_shapes
:?????????`
%
_user_specified_namegru_1_input
?<
?
'__inference_gpu_gru_with_fallback_84768

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
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:`?????????2
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
T0*H
_output_shapes6
4:`?????????:?????????: :*
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
T0*+
_output_shapes
:?????????`2
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

Identityk

Identity_1Identitytranspose_7:y:0*
T0*+
_output_shapes
:?????????`2

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

identity_3Identity_3:output:0*[
_input_shapesJ
H:?????????`:?????????:Z:Z:Z*<
api_implements*(gru_3c83e14b-7d3f-480b-af56-3855cc00958f*
api_preferred_deviceGPU*
go_backwards( *

time_major( :S O
+
_output_shapes
:?????????`
 
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
?
?
,__inference_sequential_8_layer_call_fn_85026

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
 *'
_output_shapes
:?????????0*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_sequential_8_layer_call_and_return_conditional_losses_840062
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????02

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????`:::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????`
 
_user_specified_nameinputs
?M
?
%__forward_gpu_gru_with_fallback_86452

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
api_implements*(gru_dbdfb1bc-8350-49bc-99d4-4927254488fc*
api_preferred_deviceGPU*T
backward_function_name:8__inference___backward_gpu_gru_with_fallback_86235_86453*
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
while_body_82061
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
ҭ
?
8__inference___backward_gpu_gru_with_fallback_85744_85962
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
gradients/grad_ys_0{
gradients/grad_ys_1Identityplaceholder_1*
T0*+
_output_shapes
:?????????`2
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
T0*+
_output_shapes
:`?????????*
shrink_axis_mask22
0gradients/strided_slice_grad/MLCStridedSliceGrad?
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:2.
,gradients/transpose_7_grad/InvertPermutation?
+gradients/transpose_7_grad/MLCTransposeGradMLCTransposeGradgradients/grad_ys_1:output:00gradients/transpose_7_grad/InvertPermutation:y:07gradients_transpose_7_grad_mlctransposegrad_transpose_7*
T0*+
_output_shapes
:`?????????2-
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
75loc:@gradients/strided_slice_grad/MLCStridedSliceGrad*+
_output_shapes
:`?????????2
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
T0*K
_output_shapes9
7:`?????????:?????????: :?*
rnn_modegru2*
(gradients/CudnnRNN_grad/CudnnRNNBackprop?
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:2,
*gradients/transpose_grad/InvertPermutation?
)gradients/transpose_grad/MLCTransposeGradMLCTransposeGrad9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:02gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose*
T0*+
_output_shapes
:?????????`2+
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
T0*+
_output_shapes
:?????????`2

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
?:?????????:?????????`:?????????: :`?????????:?????????::?????????`:?????????: ::`?????????:?????????: :?::?????????: :?:?:?:?:?:?::::::::::::::::::: : :: ::?*<
api_implements*(gru_dfc2bb03-a150-469d-a7e4-08432d7c3f5a*
api_preferred_deviceGPU*@
forward_function_name'%__forward_gpu_gru_with_fallback_85961*
go_backwards( *

time_major( :- )
'
_output_shapes
:?????????:1-
+
_output_shapes
:?????????`:-)
'
_output_shapes
:?????????:

_output_shapes
: :1-
+
_output_shapes
:`?????????:-)
'
_output_shapes
:?????????: 

_output_shapes
::1-
+
_output_shapes
:?????????`:1-
+
_output_shapes
:?????????:	

_output_shapes
: :


_output_shapes
::1-
+
_output_shapes
:`?????????:1-
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
ҭ
?
8__inference___backward_gpu_gru_with_fallback_80814_81032
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
gradients/grad_ys_0{
gradients/grad_ys_1Identityplaceholder_1*
T0*+
_output_shapes
:?????????`2
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
T0*+
_output_shapes
:`?????????*
shrink_axis_mask22
0gradients/strided_slice_grad/MLCStridedSliceGrad?
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:2.
,gradients/transpose_7_grad/InvertPermutation?
+gradients/transpose_7_grad/MLCTransposeGradMLCTransposeGradgradients/grad_ys_1:output:00gradients/transpose_7_grad/InvertPermutation:y:07gradients_transpose_7_grad_mlctransposegrad_transpose_7*
T0*+
_output_shapes
:`?????????2-
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
75loc:@gradients/strided_slice_grad/MLCStridedSliceGrad*+
_output_shapes
:`?????????2
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
T0*K
_output_shapes9
7:`?????????:?????????: :?*
rnn_modegru2*
(gradients/CudnnRNN_grad/CudnnRNNBackprop?
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:2,
*gradients/transpose_grad/InvertPermutation?
)gradients/transpose_grad/MLCTransposeGradMLCTransposeGrad9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:02gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose*
T0*+
_output_shapes
:?????????`2+
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
T0*+
_output_shapes
:?????????`2

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
?:?????????:?????????`:?????????: :`?????????:?????????::?????????`:?????????: ::`?????????:?????????: :?::?????????: :?:?:?:?:?:?::::::::::::::::::: : :: ::?*<
api_implements*(gru_86cc2bcf-3b62-4553-8076-900e106f9134*
api_preferred_deviceGPU*@
forward_function_name'%__forward_gpu_gru_with_fallback_81031*
go_backwards( *

time_major( :- )
'
_output_shapes
:?????????:1-
+
_output_shapes
:?????????`:-)
'
_output_shapes
:?????????:

_output_shapes
: :1-
+
_output_shapes
:`?????????:-)
'
_output_shapes
:?????????: 

_output_shapes
::1-
+
_output_shapes
:?????????`:1-
+
_output_shapes
:?????????:	

_output_shapes
: :


_output_shapes
::1-
+
_output_shapes
:`?????????:1-
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
while_body_85573
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
?;
?
__inference__traced_save_87067
file_prefix.
*savev2_dense_10_kernel_read_readvariableop,
(savev2_dense_10_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop6
2savev2_gru_1_gru_cell_2_kernel_read_readvariableop@
<savev2_gru_1_gru_cell_2_recurrent_kernel_read_readvariableop4
0savev2_gru_1_gru_cell_2_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop&
"savev2_total_2_read_readvariableop&
"savev2_count_2_read_readvariableop5
1savev2_adam_dense_10_kernel_m_read_readvariableop3
/savev2_adam_dense_10_bias_m_read_readvariableop=
9savev2_adam_gru_1_gru_cell_2_kernel_m_read_readvariableopG
Csavev2_adam_gru_1_gru_cell_2_recurrent_kernel_m_read_readvariableop;
7savev2_adam_gru_1_gru_cell_2_bias_m_read_readvariableop5
1savev2_adam_dense_10_kernel_v_read_readvariableop3
/savev2_adam_dense_10_bias_v_read_readvariableop=
9savev2_adam_gru_1_gru_cell_2_kernel_v_read_readvariableopG
Csavev2_adam_gru_1_gru_cell_2_recurrent_kernel_v_read_readvariableop;
7savev2_adam_gru_1_gru_cell_2_bias_v_read_readvariableop
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
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_10_kernel_read_readvariableop(savev2_dense_10_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop2savev2_gru_1_gru_cell_2_kernel_read_readvariableop<savev2_gru_1_gru_cell_2_recurrent_kernel_read_readvariableop0savev2_gru_1_gru_cell_2_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop"savev2_total_2_read_readvariableop"savev2_count_2_read_readvariableop1savev2_adam_dense_10_kernel_m_read_readvariableop/savev2_adam_dense_10_bias_m_read_readvariableop9savev2_adam_gru_1_gru_cell_2_kernel_m_read_readvariableopCsavev2_adam_gru_1_gru_cell_2_recurrent_kernel_m_read_readvariableop7savev2_adam_gru_1_gru_cell_2_bias_m_read_readvariableop1savev2_adam_dense_10_kernel_v_read_readvariableop/savev2_adam_dense_10_bias_v_read_readvariableop9savev2_adam_gru_1_gru_cell_2_kernel_v_read_readvariableopCsavev2_adam_gru_1_gru_cell_2_recurrent_kernel_v_read_readvariableop7savev2_adam_gru_1_gru_cell_2_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
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
?: :0:0: : : : : :Z:Z:Z: : : : : : :0:0:Z:Z:Z:0:0:Z:Z:Z: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:0: 

_output_shapes
:0:
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
: :$ 

_output_shapes

:0: 

_output_shapes
:0:$ 

_output_shapes

:Z:$ 

_output_shapes

:Z:$ 

_output_shapes

:Z:$ 

_output_shapes

:0: 

_output_shapes
:0:$ 

_output_shapes

:Z:$ 

_output_shapes

:Z:$ 

_output_shapes

:Z:

_output_shapes
: 
?
?
%__inference_gru_1_layer_call_fn_86946
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
GPU 2J 8? *I
fDRB
@__inference_gru_1_layer_call_and_return_conditional_losses_829322
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
?M
?
%__forward_gpu_gru_with_fallback_82449

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
api_implements*(gru_4ee19f71-0a49-4281-adb9-3b2ff34755fa*
api_preferred_deviceGPU*T
backward_function_name:8__inference___backward_gpu_gru_with_fallback_82232_82450*
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
while_body_84598
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
?
G__inference_sequential_8_layer_call_and_return_conditional_losses_84006

inputs
gru_1_83993
gru_1_83995
gru_1_83997
dense_10_84000
dense_10_84002
identity?? dense_10/StatefulPartitionedCall?gru_1/StatefulPartitionedCall?
gru_1/StatefulPartitionedCallStatefulPartitionedCallinputsgru_1_83993gru_1_83995gru_1_83997*
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
GPU 2J 8? *I
fDRB
@__inference_gru_1_layer_call_and_return_conditional_losses_838822
gru_1/StatefulPartitionedCall?
 dense_10/StatefulPartitionedCallStatefulPartitionedCall&gru_1/StatefulPartitionedCall:output:0dense_10_84000dense_10_84002*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????0*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_10_layer_call_and_return_conditional_losses_839232"
 dense_10/StatefulPartitionedCall?
IdentityIdentity)dense_10/StatefulPartitionedCall:output:0!^dense_10/StatefulPartitionedCall^gru_1/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????02

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????`:::::2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2>
gru_1/StatefulPartitionedCallgru_1/StatefulPartitionedCall:S O
+
_output_shapes
:?????????`
 
_user_specified_nameinputs
ڮ
?
8__inference___backward_gpu_gru_with_fallback_86235_86453
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
api_implements*(gru_dbdfb1bc-8350-49bc-99d4-4927254488fc*
api_preferred_deviceGPU*@
forward_function_name'%__forward_gpu_gru_with_fallback_86452*
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

?
C__inference_dense_10_layer_call_and_return_conditional_losses_86957

inputs%
!mlcmatmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MLCMatMul/ReadVariableOp?
MLCMatMul/ReadVariableOpReadVariableOp!mlcmatmul_readvariableop_resource*
_output_shapes

:0*
dtype02
MLCMatMul/ReadVariableOp
	MLCMatMul	MLCMatMulinputs MLCMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????02
	MLCMatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:0*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMLCMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????02	
BiasAddX
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:?????????02
Tanh?
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MLCMatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????02

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
MLCMatMul/ReadVariableOpMLCMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
G
gru_1_input8
serving_default_gru_1_input:0?????????`<
dense_100
StatefulPartitionedCall:0?????????0tensorflow/serving/predict:??
?#
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
	optimizer
	variables
regularization_losses
trainable_variables
	keras_api

signatures
*Q&call_and_return_all_conditional_losses
R__call__
S_default_save_signature"?!
_tf_keras_sequential?!{"class_name": "Sequential", "name": "sequential_8", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_8", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 96, 5]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "gru_1_input"}}, {"class_name": "GRU", "config": {"name": "gru_1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 96, 5]}, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 30, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2, "reset_after": true}}, {"class_name": "Dense", "config": {"name": "dense_10", "trainable": true, "dtype": "float32", "units": 48, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 5]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 96, 5]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_8", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 96, 5]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "gru_1_input"}}, {"class_name": "GRU", "config": {"name": "gru_1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 96, 5]}, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 30, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2, "reset_after": true}}, {"class_name": "Dense", "config": {"name": "dense_10", "trainable": true, "dtype": "float32", "units": 48, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": "mean_squared_error", "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "mae", "dtype": "float32", "fn": "mean_absolute_error"}}, {"class_name": "MeanMetricWrapper", "config": {"name": "acc", "dtype": "float32", "fn": "categorical_accuracy"}}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?
	cell


state_spec
	variables
regularization_losses
trainable_variables
	keras_api
*T&call_and_return_all_conditional_losses
U__call__"?
_tf_keras_rnn_layer?
{"class_name": "GRU", "name": "gru_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 96, 5]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "gru_1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 96, 5]}, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 30, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2, "reset_after": true}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 5]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 96, 5]}}
?

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
*V&call_and_return_all_conditional_losses
W__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_10", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_10", "trainable": true, "dtype": "float32", "units": 48, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 30}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 30]}}
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
 "
trackable_list_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
?

layers
	variables
metrics
regularization_losses
layer_metrics
trainable_variables
 non_trainable_variables
!layer_regularization_losses
R__call__
S_default_save_signature
*Q&call_and_return_all_conditional_losses
&Q"call_and_return_conditional_losses"
_generic_user_object
,
Xserving_default"
signature_map
?

kernel
recurrent_kernel
bias
"	variables
#regularization_losses
$trainable_variables
%	keras_api
*Y&call_and_return_all_conditional_losses
Z__call__"?
_tf_keras_layer?{"class_name": "GRUCell", "name": "gru_cell_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "gru_cell_2", "trainable": true, "dtype": "float32", "units": 30, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2, "reset_after": true}}
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
?

&layers

'states
	variables
(metrics
regularization_losses
)layer_metrics
trainable_variables
*non_trainable_variables
+layer_regularization_losses
U__call__
*T&call_and_return_all_conditional_losses
&T"call_and_return_conditional_losses"
_generic_user_object
!:02dense_10/kernel
:02dense_10/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?

,layers
	variables
-metrics
regularization_losses
.layer_metrics
trainable_variables
/non_trainable_variables
0layer_regularization_losses
W__call__
*V&call_and_return_all_conditional_losses
&V"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
):'Z2gru_1/gru_cell_2/kernel
3:1Z2!gru_1/gru_cell_2/recurrent_kernel
':%Z2gru_1/gru_cell_2/bias
.
0
1"
trackable_list_wrapper
5
10
21
32"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
?

4layers
"	variables
5metrics
#regularization_losses
6layer_metrics
$trainable_variables
7non_trainable_variables
8layer_regularization_losses
Z__call__
*Y&call_and_return_all_conditional_losses
&Y"call_and_return_conditional_losses"
_generic_user_object
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
 "
trackable_dict_wrapper
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
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
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
&:$02Adam/dense_10/kernel/m
 :02Adam/dense_10/bias/m
.:,Z2Adam/gru_1/gru_cell_2/kernel/m
8:6Z2(Adam/gru_1/gru_cell_2/recurrent_kernel/m
,:*Z2Adam/gru_1/gru_cell_2/bias/m
&:$02Adam/dense_10/kernel/v
 :02Adam/dense_10/bias/v
.:,Z2Adam/gru_1/gru_cell_2/kernel/v
8:6Z2(Adam/gru_1/gru_cell_2/recurrent_kernel/v
,:*Z2Adam/gru_1/gru_cell_2/bias/v
?2?
G__inference_sequential_8_layer_call_and_return_conditional_losses_83940
G__inference_sequential_8_layer_call_and_return_conditional_losses_84996
G__inference_sequential_8_layer_call_and_return_conditional_losses_84520
G__inference_sequential_8_layer_call_and_return_conditional_losses_83956?
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
,__inference_sequential_8_layer_call_fn_83988
,__inference_sequential_8_layer_call_fn_85026
,__inference_sequential_8_layer_call_fn_84019
,__inference_sequential_8_layer_call_fn_85011?
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
 __inference__wrapped_model_81041?
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
annotations? *.?+
)?&
gru_1_input?????????`
?2?
@__inference_gru_1_layer_call_and_return_conditional_losses_85495
@__inference_gru_1_layer_call_and_return_conditional_losses_86924
@__inference_gru_1_layer_call_and_return_conditional_losses_85964
@__inference_gru_1_layer_call_and_return_conditional_losses_86455?
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
?2?
%__inference_gru_1_layer_call_fn_85975
%__inference_gru_1_layer_call_fn_86935
%__inference_gru_1_layer_call_fn_86946
%__inference_gru_1_layer_call_fn_85986?
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
C__inference_dense_10_layer_call_and_return_conditional_losses_86957?
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
(__inference_dense_10_layer_call_fn_86966?
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
#__inference_signature_wrapper_84044gru_1_input"?
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
 __inference__wrapped_model_81041v8?5
.?+
)?&
gru_1_input?????????`
? "3?0
.
dense_10"?
dense_10?????????0?
C__inference_dense_10_layer_call_and_return_conditional_losses_86957\/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????0
? {
(__inference_dense_10_layer_call_fn_86966O/?,
%?"
 ?
inputs?????????
? "??????????0?
@__inference_gru_1_layer_call_and_return_conditional_losses_85495m??<
5?2
$?!
inputs?????????`

 
p

 
? "%?"
?
0?????????
? ?
@__inference_gru_1_layer_call_and_return_conditional_losses_85964m??<
5?2
$?!
inputs?????????`

 
p 

 
? "%?"
?
0?????????
? ?
@__inference_gru_1_layer_call_and_return_conditional_losses_86455}O?L
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
@__inference_gru_1_layer_call_and_return_conditional_losses_86924}O?L
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
%__inference_gru_1_layer_call_fn_85975`??<
5?2
$?!
inputs?????????`

 
p

 
? "???????????
%__inference_gru_1_layer_call_fn_85986`??<
5?2
$?!
inputs?????????`

 
p 

 
? "???????????
%__inference_gru_1_layer_call_fn_86935pO?L
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
%__inference_gru_1_layer_call_fn_86946pO?L
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
G__inference_sequential_8_layer_call_and_return_conditional_losses_83940p@?=
6?3
)?&
gru_1_input?????????`
p

 
? "%?"
?
0?????????0
? ?
G__inference_sequential_8_layer_call_and_return_conditional_losses_83956p@?=
6?3
)?&
gru_1_input?????????`
p 

 
? "%?"
?
0?????????0
? ?
G__inference_sequential_8_layer_call_and_return_conditional_losses_84520k;?8
1?.
$?!
inputs?????????`
p

 
? "%?"
?
0?????????0
? ?
G__inference_sequential_8_layer_call_and_return_conditional_losses_84996k;?8
1?.
$?!
inputs?????????`
p 

 
? "%?"
?
0?????????0
? ?
,__inference_sequential_8_layer_call_fn_83988c@?=
6?3
)?&
gru_1_input?????????`
p

 
? "??????????0?
,__inference_sequential_8_layer_call_fn_84019c@?=
6?3
)?&
gru_1_input?????????`
p 

 
? "??????????0?
,__inference_sequential_8_layer_call_fn_85011^;?8
1?.
$?!
inputs?????????`
p

 
? "??????????0?
,__inference_sequential_8_layer_call_fn_85026^;?8
1?.
$?!
inputs?????????`
p 

 
? "??????????0?
#__inference_signature_wrapper_84044?G?D
? 
=?:
8
gru_1_input)?&
gru_1_input?????????`"3?0
.
dense_10"?
dense_10?????????0