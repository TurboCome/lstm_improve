??"
??
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
E
Relu
features"T
activations"T"
Ttype:
2	
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
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?
?
While

input2T
output2T"
T
list(type)("
condfunc"
bodyfunc" 
output_shapeslist(shape)
 "
parallel_iterationsint
?"serve*	2.4.0-rc02&tf_macos-v0.1-alpha2-AS-67-gf3595294ab8??
z
dense_28/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2<* 
shared_namedense_28/kernel
s
#dense_28/kernel/Read/ReadVariableOpReadVariableOpdense_28/kernel*
_output_shapes

:2<*
dtype0
r
dense_28/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:<*
shared_namedense_28/bias
k
!dense_28/bias/Read/ReadVariableOpReadVariableOpdense_28/bias*
_output_shapes
:<*
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
&simple_rnn_8/simple_rnn_cell_16/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*7
shared_name(&simple_rnn_8/simple_rnn_cell_16/kernel
?
:simple_rnn_8/simple_rnn_cell_16/kernel/Read/ReadVariableOpReadVariableOp&simple_rnn_8/simple_rnn_cell_16/kernel*
_output_shapes
:	?*
dtype0
?
0simple_rnn_8/simple_rnn_cell_16/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*A
shared_name20simple_rnn_8/simple_rnn_cell_16/recurrent_kernel
?
Dsimple_rnn_8/simple_rnn_cell_16/recurrent_kernel/Read/ReadVariableOpReadVariableOp0simple_rnn_8/simple_rnn_cell_16/recurrent_kernel* 
_output_shapes
:
??*
dtype0
?
$simple_rnn_8/simple_rnn_cell_16/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*5
shared_name&$simple_rnn_8/simple_rnn_cell_16/bias
?
8simple_rnn_8/simple_rnn_cell_16/bias/Read/ReadVariableOpReadVariableOp$simple_rnn_8/simple_rnn_cell_16/bias*
_output_shapes	
:?*
dtype0
?
&simple_rnn_9/simple_rnn_cell_17/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?2*7
shared_name(&simple_rnn_9/simple_rnn_cell_17/kernel
?
:simple_rnn_9/simple_rnn_cell_17/kernel/Read/ReadVariableOpReadVariableOp&simple_rnn_9/simple_rnn_cell_17/kernel*
_output_shapes
:	?2*
dtype0
?
0simple_rnn_9/simple_rnn_cell_17/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:22*A
shared_name20simple_rnn_9/simple_rnn_cell_17/recurrent_kernel
?
Dsimple_rnn_9/simple_rnn_cell_17/recurrent_kernel/Read/ReadVariableOpReadVariableOp0simple_rnn_9/simple_rnn_cell_17/recurrent_kernel*
_output_shapes

:22*
dtype0
?
$simple_rnn_9/simple_rnn_cell_17/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*5
shared_name&$simple_rnn_9/simple_rnn_cell_17/bias
?
8simple_rnn_9/simple_rnn_cell_17/bias/Read/ReadVariableOpReadVariableOp$simple_rnn_9/simple_rnn_cell_17/bias*
_output_shapes
:2*
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
Adam/dense_28/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2<*'
shared_nameAdam/dense_28/kernel/m
?
*Adam/dense_28/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_28/kernel/m*
_output_shapes

:2<*
dtype0
?
Adam/dense_28/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:<*%
shared_nameAdam/dense_28/bias/m
y
(Adam/dense_28/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_28/bias/m*
_output_shapes
:<*
dtype0
?
-Adam/simple_rnn_8/simple_rnn_cell_16/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*>
shared_name/-Adam/simple_rnn_8/simple_rnn_cell_16/kernel/m
?
AAdam/simple_rnn_8/simple_rnn_cell_16/kernel/m/Read/ReadVariableOpReadVariableOp-Adam/simple_rnn_8/simple_rnn_cell_16/kernel/m*
_output_shapes
:	?*
dtype0
?
7Adam/simple_rnn_8/simple_rnn_cell_16/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*H
shared_name97Adam/simple_rnn_8/simple_rnn_cell_16/recurrent_kernel/m
?
KAdam/simple_rnn_8/simple_rnn_cell_16/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp7Adam/simple_rnn_8/simple_rnn_cell_16/recurrent_kernel/m* 
_output_shapes
:
??*
dtype0
?
+Adam/simple_rnn_8/simple_rnn_cell_16/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*<
shared_name-+Adam/simple_rnn_8/simple_rnn_cell_16/bias/m
?
?Adam/simple_rnn_8/simple_rnn_cell_16/bias/m/Read/ReadVariableOpReadVariableOp+Adam/simple_rnn_8/simple_rnn_cell_16/bias/m*
_output_shapes	
:?*
dtype0
?
-Adam/simple_rnn_9/simple_rnn_cell_17/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?2*>
shared_name/-Adam/simple_rnn_9/simple_rnn_cell_17/kernel/m
?
AAdam/simple_rnn_9/simple_rnn_cell_17/kernel/m/Read/ReadVariableOpReadVariableOp-Adam/simple_rnn_9/simple_rnn_cell_17/kernel/m*
_output_shapes
:	?2*
dtype0
?
7Adam/simple_rnn_9/simple_rnn_cell_17/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:22*H
shared_name97Adam/simple_rnn_9/simple_rnn_cell_17/recurrent_kernel/m
?
KAdam/simple_rnn_9/simple_rnn_cell_17/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp7Adam/simple_rnn_9/simple_rnn_cell_17/recurrent_kernel/m*
_output_shapes

:22*
dtype0
?
+Adam/simple_rnn_9/simple_rnn_cell_17/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*<
shared_name-+Adam/simple_rnn_9/simple_rnn_cell_17/bias/m
?
?Adam/simple_rnn_9/simple_rnn_cell_17/bias/m/Read/ReadVariableOpReadVariableOp+Adam/simple_rnn_9/simple_rnn_cell_17/bias/m*
_output_shapes
:2*
dtype0
?
Adam/dense_28/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2<*'
shared_nameAdam/dense_28/kernel/v
?
*Adam/dense_28/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_28/kernel/v*
_output_shapes

:2<*
dtype0
?
Adam/dense_28/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:<*%
shared_nameAdam/dense_28/bias/v
y
(Adam/dense_28/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_28/bias/v*
_output_shapes
:<*
dtype0
?
-Adam/simple_rnn_8/simple_rnn_cell_16/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*>
shared_name/-Adam/simple_rnn_8/simple_rnn_cell_16/kernel/v
?
AAdam/simple_rnn_8/simple_rnn_cell_16/kernel/v/Read/ReadVariableOpReadVariableOp-Adam/simple_rnn_8/simple_rnn_cell_16/kernel/v*
_output_shapes
:	?*
dtype0
?
7Adam/simple_rnn_8/simple_rnn_cell_16/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*H
shared_name97Adam/simple_rnn_8/simple_rnn_cell_16/recurrent_kernel/v
?
KAdam/simple_rnn_8/simple_rnn_cell_16/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp7Adam/simple_rnn_8/simple_rnn_cell_16/recurrent_kernel/v* 
_output_shapes
:
??*
dtype0
?
+Adam/simple_rnn_8/simple_rnn_cell_16/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*<
shared_name-+Adam/simple_rnn_8/simple_rnn_cell_16/bias/v
?
?Adam/simple_rnn_8/simple_rnn_cell_16/bias/v/Read/ReadVariableOpReadVariableOp+Adam/simple_rnn_8/simple_rnn_cell_16/bias/v*
_output_shapes	
:?*
dtype0
?
-Adam/simple_rnn_9/simple_rnn_cell_17/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?2*>
shared_name/-Adam/simple_rnn_9/simple_rnn_cell_17/kernel/v
?
AAdam/simple_rnn_9/simple_rnn_cell_17/kernel/v/Read/ReadVariableOpReadVariableOp-Adam/simple_rnn_9/simple_rnn_cell_17/kernel/v*
_output_shapes
:	?2*
dtype0
?
7Adam/simple_rnn_9/simple_rnn_cell_17/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:22*H
shared_name97Adam/simple_rnn_9/simple_rnn_cell_17/recurrent_kernel/v
?
KAdam/simple_rnn_9/simple_rnn_cell_17/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp7Adam/simple_rnn_9/simple_rnn_cell_17/recurrent_kernel/v*
_output_shapes

:22*
dtype0
?
+Adam/simple_rnn_9/simple_rnn_cell_17/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*<
shared_name-+Adam/simple_rnn_9/simple_rnn_cell_17/bias/v
?
?Adam/simple_rnn_9/simple_rnn_cell_17/bias/v/Read/ReadVariableOpReadVariableOp+Adam/simple_rnn_9/simple_rnn_cell_17/bias/v*
_output_shapes
:2*
dtype0

NoOpNoOp
?>
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?>
value?>B?> B?>
?
layer_with_weights-0
layer-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer-5
layer_with_weights-2
layer-6
	optimizer
	regularization_losses

	variables
trainable_variables
	keras_api

signatures
l
cell

state_spec
regularization_losses
	variables
trainable_variables
	keras_api
R
regularization_losses
	variables
trainable_variables
	keras_api
R
regularization_losses
	variables
trainable_variables
	keras_api
l
cell

state_spec
regularization_losses
	variables
 trainable_variables
!	keras_api
R
"regularization_losses
#	variables
$trainable_variables
%	keras_api
R
&regularization_losses
'	variables
(trainable_variables
)	keras_api
h

*kernel
+bias
,regularization_losses
-	variables
.trainable_variables
/	keras_api
?
0iter

1beta_1

2beta_2
	3decay
4learning_rate*m?+m?5m?6m?7m?8m?9m?:m?*v?+v?5v?6v?7v?8v?9v?:v?
 
8
50
61
72
83
94
:5
*6
+7
8
50
61
72
83
94
:5
*6
+7
?
	regularization_losses
;layer_regularization_losses
<layer_metrics

	variables
=non_trainable_variables
trainable_variables
>metrics

?layers
 
~

5kernel
6recurrent_kernel
7bias
@regularization_losses
A	variables
Btrainable_variables
C	keras_api
 
 

50
61
72

50
61
72
?
regularization_losses
Dlayer_regularization_losses
Elayer_metrics
	variables
Fnon_trainable_variables
trainable_variables
Gmetrics

Hlayers

Istates
 
 
 
?
regularization_losses
Jlayer_regularization_losses
Klayer_metrics
	variables
Lnon_trainable_variables
trainable_variables
Mmetrics

Nlayers
 
 
 
?
regularization_losses
Olayer_regularization_losses
Player_metrics
	variables
Qnon_trainable_variables
trainable_variables
Rmetrics

Slayers
~

8kernel
9recurrent_kernel
:bias
Tregularization_losses
U	variables
Vtrainable_variables
W	keras_api
 
 

80
91
:2

80
91
:2
?
regularization_losses
Xlayer_regularization_losses
Ylayer_metrics
	variables
Znon_trainable_variables
 trainable_variables
[metrics

\layers

]states
 
 
 
?
"regularization_losses
^layer_regularization_losses
_layer_metrics
#	variables
`non_trainable_variables
$trainable_variables
ametrics

blayers
 
 
 
?
&regularization_losses
clayer_regularization_losses
dlayer_metrics
'	variables
enon_trainable_variables
(trainable_variables
fmetrics

glayers
[Y
VARIABLE_VALUEdense_28/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_28/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

*0
+1

*0
+1
?
,regularization_losses
hlayer_regularization_losses
ilayer_metrics
-	variables
jnon_trainable_variables
.trainable_variables
kmetrics

llayers
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
b`
VARIABLE_VALUE&simple_rnn_8/simple_rnn_cell_16/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUE0simple_rnn_8/simple_rnn_cell_16/recurrent_kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUE$simple_rnn_8/simple_rnn_cell_16/bias&variables/2/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUE&simple_rnn_9/simple_rnn_cell_17/kernel&variables/3/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUE0simple_rnn_9/simple_rnn_cell_17/recurrent_kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUE$simple_rnn_9/simple_rnn_cell_17/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE
 
 
 

m0
n1
o2
1
0
1
2
3
4
5
6
 

50
61
72

50
61
72
?
@regularization_losses
player_regularization_losses
qlayer_metrics
A	variables
rnon_trainable_variables
Btrainable_variables
smetrics

tlayers
 
 
 
 

0
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
 
 

80
91
:2

80
91
:2
?
Tregularization_losses
ulayer_regularization_losses
vlayer_metrics
U	variables
wnon_trainable_variables
Vtrainable_variables
xmetrics

ylayers
 
 
 
 

0
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
 
 
 
 
 
 
4
	ztotal
	{count
|	variables
}	keras_api
G
	~total
	count
?
_fn_kwargs
?	variables
?	keras_api
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api
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
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

z0
{1

|	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

~0
1

?	variables
QO
VARIABLE_VALUEtotal_24keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_24keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
~|
VARIABLE_VALUEAdam/dense_28/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_28/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE-Adam/simple_rnn_8/simple_rnn_cell_16/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE7Adam/simple_rnn_8/simple_rnn_cell_16/recurrent_kernel/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE+Adam/simple_rnn_8/simple_rnn_cell_16/bias/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE-Adam/simple_rnn_9/simple_rnn_cell_17/kernel/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE7Adam/simple_rnn_9/simple_rnn_cell_17/recurrent_kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE+Adam/simple_rnn_9/simple_rnn_cell_17/bias/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_28/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_28/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE-Adam/simple_rnn_8/simple_rnn_cell_16/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE7Adam/simple_rnn_8/simple_rnn_cell_16/recurrent_kernel/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE+Adam/simple_rnn_8/simple_rnn_cell_16/bias/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE-Adam/simple_rnn_9/simple_rnn_cell_17/kernel/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE7Adam/simple_rnn_9/simple_rnn_cell_17/recurrent_kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE+Adam/simple_rnn_9/simple_rnn_cell_17/bias/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
"serving_default_simple_rnn_8_inputPlaceholder*+
_output_shapes
:?????????x*
dtype0* 
shape:?????????x
?
StatefulPartitionedCallStatefulPartitionedCall"serving_default_simple_rnn_8_input&simple_rnn_8/simple_rnn_cell_16/kernel$simple_rnn_8/simple_rnn_cell_16/bias0simple_rnn_8/simple_rnn_cell_16/recurrent_kernel&simple_rnn_9/simple_rnn_cell_17/kernel$simple_rnn_9/simple_rnn_cell_17/bias0simple_rnn_9/simple_rnn_cell_17/recurrent_kerneldense_28/kerneldense_28/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????<**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *-
f(R&
$__inference_signature_wrapper_195262
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_28/kernel/Read/ReadVariableOp!dense_28/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp:simple_rnn_8/simple_rnn_cell_16/kernel/Read/ReadVariableOpDsimple_rnn_8/simple_rnn_cell_16/recurrent_kernel/Read/ReadVariableOp8simple_rnn_8/simple_rnn_cell_16/bias/Read/ReadVariableOp:simple_rnn_9/simple_rnn_cell_17/kernel/Read/ReadVariableOpDsimple_rnn_9/simple_rnn_cell_17/recurrent_kernel/Read/ReadVariableOp8simple_rnn_9/simple_rnn_cell_17/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal_2/Read/ReadVariableOpcount_2/Read/ReadVariableOp*Adam/dense_28/kernel/m/Read/ReadVariableOp(Adam/dense_28/bias/m/Read/ReadVariableOpAAdam/simple_rnn_8/simple_rnn_cell_16/kernel/m/Read/ReadVariableOpKAdam/simple_rnn_8/simple_rnn_cell_16/recurrent_kernel/m/Read/ReadVariableOp?Adam/simple_rnn_8/simple_rnn_cell_16/bias/m/Read/ReadVariableOpAAdam/simple_rnn_9/simple_rnn_cell_17/kernel/m/Read/ReadVariableOpKAdam/simple_rnn_9/simple_rnn_cell_17/recurrent_kernel/m/Read/ReadVariableOp?Adam/simple_rnn_9/simple_rnn_cell_17/bias/m/Read/ReadVariableOp*Adam/dense_28/kernel/v/Read/ReadVariableOp(Adam/dense_28/bias/v/Read/ReadVariableOpAAdam/simple_rnn_8/simple_rnn_cell_16/kernel/v/Read/ReadVariableOpKAdam/simple_rnn_8/simple_rnn_cell_16/recurrent_kernel/v/Read/ReadVariableOp?Adam/simple_rnn_8/simple_rnn_cell_16/bias/v/Read/ReadVariableOpAAdam/simple_rnn_9/simple_rnn_cell_17/kernel/v/Read/ReadVariableOpKAdam/simple_rnn_9/simple_rnn_cell_17/recurrent_kernel/v/Read/ReadVariableOp?Adam/simple_rnn_9/simple_rnn_cell_17/bias/v/Read/ReadVariableOpConst*0
Tin)
'2%	*
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
GPU 2J 8? *(
f#R!
__inference__traced_save_197098
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_28/kerneldense_28/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rate&simple_rnn_8/simple_rnn_cell_16/kernel0simple_rnn_8/simple_rnn_cell_16/recurrent_kernel$simple_rnn_8/simple_rnn_cell_16/bias&simple_rnn_9/simple_rnn_cell_17/kernel0simple_rnn_9/simple_rnn_cell_17/recurrent_kernel$simple_rnn_9/simple_rnn_cell_17/biastotalcounttotal_1count_1total_2count_2Adam/dense_28/kernel/mAdam/dense_28/bias/m-Adam/simple_rnn_8/simple_rnn_cell_16/kernel/m7Adam/simple_rnn_8/simple_rnn_cell_16/recurrent_kernel/m+Adam/simple_rnn_8/simple_rnn_cell_16/bias/m-Adam/simple_rnn_9/simple_rnn_cell_17/kernel/m7Adam/simple_rnn_9/simple_rnn_cell_17/recurrent_kernel/m+Adam/simple_rnn_9/simple_rnn_cell_17/bias/mAdam/dense_28/kernel/vAdam/dense_28/bias/v-Adam/simple_rnn_8/simple_rnn_cell_16/kernel/v7Adam/simple_rnn_8/simple_rnn_cell_16/recurrent_kernel/v+Adam/simple_rnn_8/simple_rnn_cell_16/bias/v-Adam/simple_rnn_9/simple_rnn_cell_17/kernel/v7Adam/simple_rnn_9/simple_rnn_cell_17/recurrent_kernel/v+Adam/simple_rnn_9/simple_rnn_cell_17/bias/v*/
Tin(
&2$*
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
GPU 2J 8? *+
f&R$
"__inference__traced_restore_197213??
?
?
N__inference_simple_rnn_cell_16_layer_call_and_return_conditional_losses_193531

inputs

states"
matmul_readvariableop_resource#
biasadd_readvariableop_resource$
 matmul_1_readvariableop_resource
identity

identity_1??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?MatMul_1/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOpw
MatMul	MLCMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAdd?
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul_1/ReadVariableOp}
MatMul_1	MLCMatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2

MatMul_1l
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
addP
TanhTanhadd:z:0*
T0*(
_output_shapes
:??????????2
Tanh?
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity?

Identity_1IdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*F
_input_shapes5
3:?????????:??????????:::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:PL
(
_output_shapes
:??????????
 
_user_specified_namestates
?
?
N__inference_simple_rnn_cell_17_layer_call_and_return_conditional_losses_196942

inputs
states_0"
matmul_readvariableop_resource#
biasadd_readvariableop_resource$
 matmul_1_readvariableop_resource
identity

identity_1??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?MatMul_1/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?2*
dtype02
MatMul/ReadVariableOpv
MatMul	MLCMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:2*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22	
BiasAdd?
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:22*
dtype02
MatMul_1/ReadVariableOp~
MatMul_1	MLCMatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22

MatMul_1k
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*'
_output_shapes
:?????????22
addO
TanhTanhadd:z:0*
T0*'
_output_shapes
:?????????22
Tanh?
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????22

Identity?

Identity_1IdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????22

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*F
_input_shapes5
3:??????????:?????????2:::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:QM
'
_output_shapes
:?????????2
"
_user_specified_name
states/0
?C
?
simple_rnn_8_while_body_1955476
2simple_rnn_8_while_simple_rnn_8_while_loop_counter<
8simple_rnn_8_while_simple_rnn_8_while_maximum_iterations"
simple_rnn_8_while_placeholder$
 simple_rnn_8_while_placeholder_1$
 simple_rnn_8_while_placeholder_25
1simple_rnn_8_while_simple_rnn_8_strided_slice_1_0q
msimple_rnn_8_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_8_tensorarrayunstack_tensorlistfromtensor_0J
Fsimple_rnn_8_while_simple_rnn_cell_16_matmul_readvariableop_resource_0K
Gsimple_rnn_8_while_simple_rnn_cell_16_biasadd_readvariableop_resource_0L
Hsimple_rnn_8_while_simple_rnn_cell_16_matmul_1_readvariableop_resource_0
simple_rnn_8_while_identity!
simple_rnn_8_while_identity_1!
simple_rnn_8_while_identity_2!
simple_rnn_8_while_identity_3!
simple_rnn_8_while_identity_43
/simple_rnn_8_while_simple_rnn_8_strided_slice_1o
ksimple_rnn_8_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_8_tensorarrayunstack_tensorlistfromtensorH
Dsimple_rnn_8_while_simple_rnn_cell_16_matmul_readvariableop_resourceI
Esimple_rnn_8_while_simple_rnn_cell_16_biasadd_readvariableop_resourceJ
Fsimple_rnn_8_while_simple_rnn_cell_16_matmul_1_readvariableop_resource??<simple_rnn_8/while/simple_rnn_cell_16/BiasAdd/ReadVariableOp?;simple_rnn_8/while/simple_rnn_cell_16/MatMul/ReadVariableOp?=simple_rnn_8/while/simple_rnn_cell_16/MatMul_1/ReadVariableOp?
Dsimple_rnn_8/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2F
Dsimple_rnn_8/while/TensorArrayV2Read/TensorListGetItem/element_shape?
6simple_rnn_8/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemmsimple_rnn_8_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_8_tensorarrayunstack_tensorlistfromtensor_0simple_rnn_8_while_placeholderMsimple_rnn_8/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype028
6simple_rnn_8/while/TensorArrayV2Read/TensorListGetItem?
;simple_rnn_8/while/simple_rnn_cell_16/MatMul/ReadVariableOpReadVariableOpFsimple_rnn_8_while_simple_rnn_cell_16_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype02=
;simple_rnn_8/while/simple_rnn_cell_16/MatMul/ReadVariableOp?
,simple_rnn_8/while/simple_rnn_cell_16/MatMul	MLCMatMul=simple_rnn_8/while/TensorArrayV2Read/TensorListGetItem:item:0Csimple_rnn_8/while/simple_rnn_cell_16/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2.
,simple_rnn_8/while/simple_rnn_cell_16/MatMul?
<simple_rnn_8/while/simple_rnn_cell_16/BiasAdd/ReadVariableOpReadVariableOpGsimple_rnn_8_while_simple_rnn_cell_16_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02>
<simple_rnn_8/while/simple_rnn_cell_16/BiasAdd/ReadVariableOp?
-simple_rnn_8/while/simple_rnn_cell_16/BiasAddBiasAdd6simple_rnn_8/while/simple_rnn_cell_16/MatMul:product:0Dsimple_rnn_8/while/simple_rnn_cell_16/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2/
-simple_rnn_8/while/simple_rnn_cell_16/BiasAdd?
=simple_rnn_8/while/simple_rnn_cell_16/MatMul_1/ReadVariableOpReadVariableOpHsimple_rnn_8_while_simple_rnn_cell_16_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02?
=simple_rnn_8/while/simple_rnn_cell_16/MatMul_1/ReadVariableOp?
.simple_rnn_8/while/simple_rnn_cell_16/MatMul_1	MLCMatMul simple_rnn_8_while_placeholder_2Esimple_rnn_8/while/simple_rnn_cell_16/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????20
.simple_rnn_8/while/simple_rnn_cell_16/MatMul_1?
)simple_rnn_8/while/simple_rnn_cell_16/addAddV26simple_rnn_8/while/simple_rnn_cell_16/BiasAdd:output:08simple_rnn_8/while/simple_rnn_cell_16/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2+
)simple_rnn_8/while/simple_rnn_cell_16/add?
*simple_rnn_8/while/simple_rnn_cell_16/TanhTanh-simple_rnn_8/while/simple_rnn_cell_16/add:z:0*
T0*(
_output_shapes
:??????????2,
*simple_rnn_8/while/simple_rnn_cell_16/Tanh?
7simple_rnn_8/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem simple_rnn_8_while_placeholder_1simple_rnn_8_while_placeholder.simple_rnn_8/while/simple_rnn_cell_16/Tanh:y:0*
_output_shapes
: *
element_dtype029
7simple_rnn_8/while/TensorArrayV2Write/TensorListSetItemv
simple_rnn_8/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
simple_rnn_8/while/add/y?
simple_rnn_8/while/addAddV2simple_rnn_8_while_placeholder!simple_rnn_8/while/add/y:output:0*
T0*
_output_shapes
: 2
simple_rnn_8/while/addz
simple_rnn_8/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
simple_rnn_8/while/add_1/y?
simple_rnn_8/while/add_1AddV22simple_rnn_8_while_simple_rnn_8_while_loop_counter#simple_rnn_8/while/add_1/y:output:0*
T0*
_output_shapes
: 2
simple_rnn_8/while/add_1?
simple_rnn_8/while/IdentityIdentitysimple_rnn_8/while/add_1:z:0=^simple_rnn_8/while/simple_rnn_cell_16/BiasAdd/ReadVariableOp<^simple_rnn_8/while/simple_rnn_cell_16/MatMul/ReadVariableOp>^simple_rnn_8/while/simple_rnn_cell_16/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
simple_rnn_8/while/Identity?
simple_rnn_8/while/Identity_1Identity8simple_rnn_8_while_simple_rnn_8_while_maximum_iterations=^simple_rnn_8/while/simple_rnn_cell_16/BiasAdd/ReadVariableOp<^simple_rnn_8/while/simple_rnn_cell_16/MatMul/ReadVariableOp>^simple_rnn_8/while/simple_rnn_cell_16/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
simple_rnn_8/while/Identity_1?
simple_rnn_8/while/Identity_2Identitysimple_rnn_8/while/add:z:0=^simple_rnn_8/while/simple_rnn_cell_16/BiasAdd/ReadVariableOp<^simple_rnn_8/while/simple_rnn_cell_16/MatMul/ReadVariableOp>^simple_rnn_8/while/simple_rnn_cell_16/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
simple_rnn_8/while/Identity_2?
simple_rnn_8/while/Identity_3IdentityGsimple_rnn_8/while/TensorArrayV2Write/TensorListSetItem:output_handle:0=^simple_rnn_8/while/simple_rnn_cell_16/BiasAdd/ReadVariableOp<^simple_rnn_8/while/simple_rnn_cell_16/MatMul/ReadVariableOp>^simple_rnn_8/while/simple_rnn_cell_16/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
simple_rnn_8/while/Identity_3?
simple_rnn_8/while/Identity_4Identity.simple_rnn_8/while/simple_rnn_cell_16/Tanh:y:0=^simple_rnn_8/while/simple_rnn_cell_16/BiasAdd/ReadVariableOp<^simple_rnn_8/while/simple_rnn_cell_16/MatMul/ReadVariableOp>^simple_rnn_8/while/simple_rnn_cell_16/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2
simple_rnn_8/while/Identity_4"C
simple_rnn_8_while_identity$simple_rnn_8/while/Identity:output:0"G
simple_rnn_8_while_identity_1&simple_rnn_8/while/Identity_1:output:0"G
simple_rnn_8_while_identity_2&simple_rnn_8/while/Identity_2:output:0"G
simple_rnn_8_while_identity_3&simple_rnn_8/while/Identity_3:output:0"G
simple_rnn_8_while_identity_4&simple_rnn_8/while/Identity_4:output:0"d
/simple_rnn_8_while_simple_rnn_8_strided_slice_11simple_rnn_8_while_simple_rnn_8_strided_slice_1_0"?
Esimple_rnn_8_while_simple_rnn_cell_16_biasadd_readvariableop_resourceGsimple_rnn_8_while_simple_rnn_cell_16_biasadd_readvariableop_resource_0"?
Fsimple_rnn_8_while_simple_rnn_cell_16_matmul_1_readvariableop_resourceHsimple_rnn_8_while_simple_rnn_cell_16_matmul_1_readvariableop_resource_0"?
Dsimple_rnn_8_while_simple_rnn_cell_16_matmul_readvariableop_resourceFsimple_rnn_8_while_simple_rnn_cell_16_matmul_readvariableop_resource_0"?
ksimple_rnn_8_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_8_tensorarrayunstack_tensorlistfromtensormsimple_rnn_8_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_8_tensorarrayunstack_tensorlistfromtensor_0*?
_input_shapes.
,: : : : :??????????: : :::2|
<simple_rnn_8/while/simple_rnn_cell_16/BiasAdd/ReadVariableOp<simple_rnn_8/while/simple_rnn_cell_16/BiasAdd/ReadVariableOp2z
;simple_rnn_8/while/simple_rnn_cell_16/MatMul/ReadVariableOp;simple_rnn_8/while/simple_rnn_cell_16/MatMul/ReadVariableOp2~
=simple_rnn_8/while/simple_rnn_cell_16/MatMul_1/ReadVariableOp=simple_rnn_8/while/simple_rnn_cell_16/MatMul_1/ReadVariableOp: 
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
: :.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?
c
E__inference_dropout_8_layer_call_and_return_conditional_losses_194773

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:?????????x?2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:?????????x?2

Identity_1"!

identity_1Identity_1:output:0*+
_input_shapes
:?????????x?:T P
,
_output_shapes
:?????????x?
 
_user_specified_nameinputs
?=
?
H__inference_simple_rnn_8_layer_call_and_return_conditional_losses_193968

inputs
simple_rnn_cell_16_193893
simple_rnn_cell_16_193895
simple_rnn_cell_16_193897
identity??*simple_rnn_cell_16/StatefulPartitionedCall?whileD
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
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
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

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
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
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
zerosu
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
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
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
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
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
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_2?
*simple_rnn_cell_16/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0simple_rnn_cell_16_193893simple_rnn_cell_16_193895simple_rnn_cell_16_193897*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:??????????:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_simple_rnn_cell_16_layer_call_and_return_conditional_losses_1935312,
*simple_rnn_cell_16/StatefulPartitionedCall?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
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
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0simple_rnn_cell_16_193893simple_rnn_cell_16_193895simple_rnn_cell_16_193897*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :??????????: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_193905*
condR
while_cond_193904*9
output_shapes(
&: : : : :??????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:???????????????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:???????????????????2
transpose_1?
IdentityIdentitytranspose_1:y:0+^simple_rnn_cell_16/StatefulPartitionedCall^while*
T0*5
_output_shapes#
!:???????????????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????:::2X
*simple_rnn_cell_16/StatefulPartitionedCall*simple_rnn_cell_16/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?#
?
while_body_194417
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0%
!while_simple_rnn_cell_17_194439_0%
!while_simple_rnn_cell_17_194441_0%
!while_simple_rnn_cell_17_194443_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor#
while_simple_rnn_cell_17_194439#
while_simple_rnn_cell_17_194441#
while_simple_rnn_cell_17_194443??0while/simple_rnn_cell_17/StatefulPartitionedCall?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
0while/simple_rnn_cell_17/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2!while_simple_rnn_cell_17_194439_0!while_simple_rnn_cell_17_194441_0!while_simple_rnn_cell_17_194443_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????2:?????????2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_simple_rnn_cell_17_layer_call_and_return_conditional_losses_19404322
0while/simple_rnn_cell_17/StatefulPartitionedCall?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder9while/simple_rnn_cell_17/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1?
while/IdentityIdentitywhile/add_1:z:01^while/simple_rnn_cell_17/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations1^while/simple_rnn_cell_17/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:01^while/simple_rnn_cell_17/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:01^while/simple_rnn_cell_17/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity9while/simple_rnn_cell_17/StatefulPartitionedCall:output:11^while/simple_rnn_cell_17/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????22
while/Identity_4")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"D
while_simple_rnn_cell_17_194439!while_simple_rnn_cell_17_194439_0"D
while_simple_rnn_cell_17_194441!while_simple_rnn_cell_17_194441_0"D
while_simple_rnn_cell_17_194443!while_simple_rnn_cell_17_194443_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :?????????2: : :::2d
0while/simple_rnn_cell_17/StatefulPartitionedCall0while/simple_rnn_cell_17/StatefulPartitionedCall: 
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
:?????????2:

_output_shapes
: :

_output_shapes
: 
?
e
I__inference_activation_20_layer_call_and_return_conditional_losses_194751

inputs
identityS
ReluReluinputs*
T0*,
_output_shapes
:?????????x?2
Reluk
IdentityIdentityRelu:activations:0*
T0*,
_output_shapes
:?????????x?2

Identity"
identityIdentity:output:0*+
_input_shapes
:?????????x?:T P
,
_output_shapes
:?????????x?
 
_user_specified_nameinputs
?H
?
H__inference_simple_rnn_9_layer_call_and_return_conditional_losses_196412

inputs5
1simple_rnn_cell_17_matmul_readvariableop_resource6
2simple_rnn_cell_17_biasadd_readvariableop_resource7
3simple_rnn_cell_17_matmul_1_readvariableop_resource
identity??)simple_rnn_cell_17/BiasAdd/ReadVariableOp?(simple_rnn_cell_17/MatMul/ReadVariableOp?*simple_rnn_cell_17/MatMul_1/ReadVariableOp?whileD
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
value	B :22
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
value	B :22
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
:?????????22
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm{
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:x??????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
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
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_2?
(simple_rnn_cell_17/MatMul/ReadVariableOpReadVariableOp1simple_rnn_cell_17_matmul_readvariableop_resource*
_output_shapes
:	?2*
dtype02*
(simple_rnn_cell_17/MatMul/ReadVariableOp?
simple_rnn_cell_17/MatMul	MLCMatMulstrided_slice_2:output:00simple_rnn_cell_17/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
simple_rnn_cell_17/MatMul?
)simple_rnn_cell_17/BiasAdd/ReadVariableOpReadVariableOp2simple_rnn_cell_17_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02+
)simple_rnn_cell_17/BiasAdd/ReadVariableOp?
simple_rnn_cell_17/BiasAddBiasAdd#simple_rnn_cell_17/MatMul:product:01simple_rnn_cell_17/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
simple_rnn_cell_17/BiasAdd?
*simple_rnn_cell_17/MatMul_1/ReadVariableOpReadVariableOp3simple_rnn_cell_17_matmul_1_readvariableop_resource*
_output_shapes

:22*
dtype02,
*simple_rnn_cell_17/MatMul_1/ReadVariableOp?
simple_rnn_cell_17/MatMul_1	MLCMatMulzeros:output:02simple_rnn_cell_17/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
simple_rnn_cell_17/MatMul_1?
simple_rnn_cell_17/addAddV2#simple_rnn_cell_17/BiasAdd:output:0%simple_rnn_cell_17/MatMul_1:product:0*
T0*'
_output_shapes
:?????????22
simple_rnn_cell_17/add?
simple_rnn_cell_17/TanhTanhsimple_rnn_cell_17/add:z:0*
T0*'
_output_shapes
:?????????22
simple_rnn_cell_17/Tanh?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:01simple_rnn_cell_17_matmul_readvariableop_resource2simple_rnn_cell_17_biasadd_readvariableop_resource3simple_rnn_cell_17_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????2: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_196346*
condR
while_cond_196345*8
output_shapes'
%: : : : :?????????2: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:x?????????2*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????2*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????x22
transpose_1?
IdentityIdentitystrided_slice_3:output:0*^simple_rnn_cell_17/BiasAdd/ReadVariableOp)^simple_rnn_cell_17/MatMul/ReadVariableOp+^simple_rnn_cell_17/MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????x?:::2V
)simple_rnn_cell_17/BiasAdd/ReadVariableOp)simple_rnn_cell_17/BiasAdd/ReadVariableOp2T
(simple_rnn_cell_17/MatMul/ReadVariableOp(simple_rnn_cell_17/MatMul/ReadVariableOp2X
*simple_rnn_cell_17/MatMul_1/ReadVariableOp*simple_rnn_cell_17/MatMul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:?????????x?
 
_user_specified_nameinputs
?H
?
H__inference_simple_rnn_8_layer_call_and_return_conditional_losses_194716

inputs5
1simple_rnn_cell_16_matmul_readvariableop_resource6
2simple_rnn_cell_16_biasadd_readvariableop_resource7
3simple_rnn_cell_16_matmul_1_readvariableop_resource
identity??)simple_rnn_cell_16/BiasAdd/ReadVariableOp?(simple_rnn_cell_16/MatMul/ReadVariableOp?*simple_rnn_cell_16/MatMul_1/ReadVariableOp?whileD
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
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
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

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
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
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:x?????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
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
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
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
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_2?
(simple_rnn_cell_16/MatMul/ReadVariableOpReadVariableOp1simple_rnn_cell_16_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02*
(simple_rnn_cell_16/MatMul/ReadVariableOp?
simple_rnn_cell_16/MatMul	MLCMatMulstrided_slice_2:output:00simple_rnn_cell_16/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
simple_rnn_cell_16/MatMul?
)simple_rnn_cell_16/BiasAdd/ReadVariableOpReadVariableOp2simple_rnn_cell_16_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)simple_rnn_cell_16/BiasAdd/ReadVariableOp?
simple_rnn_cell_16/BiasAddBiasAdd#simple_rnn_cell_16/MatMul:product:01simple_rnn_cell_16/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
simple_rnn_cell_16/BiasAdd?
*simple_rnn_cell_16/MatMul_1/ReadVariableOpReadVariableOp3simple_rnn_cell_16_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02,
*simple_rnn_cell_16/MatMul_1/ReadVariableOp?
simple_rnn_cell_16/MatMul_1	MLCMatMulzeros:output:02simple_rnn_cell_16/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
simple_rnn_cell_16/MatMul_1?
simple_rnn_cell_16/addAddV2#simple_rnn_cell_16/BiasAdd:output:0%simple_rnn_cell_16/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
simple_rnn_cell_16/add?
simple_rnn_cell_16/TanhTanhsimple_rnn_cell_16/add:z:0*
T0*(
_output_shapes
:??????????2
simple_rnn_cell_16/Tanh?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:01simple_rnn_cell_16_matmul_readvariableop_resource2simple_rnn_cell_16_biasadd_readvariableop_resource3simple_rnn_cell_16_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :??????????: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_194650*
condR
while_cond_194649*9
output_shapes(
&: : : : :??????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:x??????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:?????????x?2
transpose_1?
IdentityIdentitytranspose_1:y:0*^simple_rnn_cell_16/BiasAdd/ReadVariableOp)^simple_rnn_cell_16/MatMul/ReadVariableOp+^simple_rnn_cell_16/MatMul_1/ReadVariableOp^while*
T0*,
_output_shapes
:?????????x?2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????x:::2V
)simple_rnn_cell_16/BiasAdd/ReadVariableOp)simple_rnn_cell_16/BiasAdd/ReadVariableOp2T
(simple_rnn_cell_16/MatMul/ReadVariableOp(simple_rnn_cell_16/MatMul/ReadVariableOp2X
*simple_rnn_cell_16/MatMul_1/ReadVariableOp*simple_rnn_cell_16/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:?????????x
 
_user_specified_nameinputs
?
?
while_cond_196591
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_196591___redundant_placeholder04
0while_while_cond_196591___redundant_placeholder14
0while_while_cond_196591___redundant_placeholder24
0while_while_cond_196591___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
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
while_identitywhile/Identity:output:0*@
_input_shapes/
-: : : : :?????????2: ::::: 
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
:?????????2:

_output_shapes
: :

_output_shapes
:
?
a
E__inference_dropout_8_layer_call_and_return_conditional_losses_196285

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout/Constx
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:?????????x?2
dropout/Mula
dropout/rateConst*
_output_shapes
: *
dtype0*
valueB
 *???=2
dropout/rate^
dropout/seedConst*
_output_shapes
: *
dtype0*
value	B : 2
dropout/seed?
dropout
MLCDropoutinputsdropout/rate:output:0dropout/seed:output:0*
T0*,
_output_shapes
:?????????x?2	
dropouti
IdentityIdentitydropout:output:0*
T0*,
_output_shapes
:?????????x?2

Identity"
identityIdentity:output:0*+
_input_shapes
:?????????x?:T P
,
_output_shapes
:?????????x?
 
_user_specified_nameinputs
?4
?
while_body_194538
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0=
9while_simple_rnn_cell_16_matmul_readvariableop_resource_0>
:while_simple_rnn_cell_16_biasadd_readvariableop_resource_0?
;while_simple_rnn_cell_16_matmul_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor;
7while_simple_rnn_cell_16_matmul_readvariableop_resource<
8while_simple_rnn_cell_16_biasadd_readvariableop_resource=
9while_simple_rnn_cell_16_matmul_1_readvariableop_resource??/while/simple_rnn_cell_16/BiasAdd/ReadVariableOp?.while/simple_rnn_cell_16/MatMul/ReadVariableOp?0while/simple_rnn_cell_16/MatMul_1/ReadVariableOp?
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
.while/simple_rnn_cell_16/MatMul/ReadVariableOpReadVariableOp9while_simple_rnn_cell_16_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype020
.while/simple_rnn_cell_16/MatMul/ReadVariableOp?
while/simple_rnn_cell_16/MatMul	MLCMatMul0while/TensorArrayV2Read/TensorListGetItem:item:06while/simple_rnn_cell_16/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2!
while/simple_rnn_cell_16/MatMul?
/while/simple_rnn_cell_16/BiasAdd/ReadVariableOpReadVariableOp:while_simple_rnn_cell_16_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype021
/while/simple_rnn_cell_16/BiasAdd/ReadVariableOp?
 while/simple_rnn_cell_16/BiasAddBiasAdd)while/simple_rnn_cell_16/MatMul:product:07while/simple_rnn_cell_16/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2"
 while/simple_rnn_cell_16/BiasAdd?
0while/simple_rnn_cell_16/MatMul_1/ReadVariableOpReadVariableOp;while_simple_rnn_cell_16_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype022
0while/simple_rnn_cell_16/MatMul_1/ReadVariableOp?
!while/simple_rnn_cell_16/MatMul_1	MLCMatMulwhile_placeholder_28while/simple_rnn_cell_16/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2#
!while/simple_rnn_cell_16/MatMul_1?
while/simple_rnn_cell_16/addAddV2)while/simple_rnn_cell_16/BiasAdd:output:0+while/simple_rnn_cell_16/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
while/simple_rnn_cell_16/add?
while/simple_rnn_cell_16/TanhTanh while/simple_rnn_cell_16/add:z:0*
T0*(
_output_shapes
:??????????2
while/simple_rnn_cell_16/Tanh?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder!while/simple_rnn_cell_16/Tanh:y:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1?
while/IdentityIdentitywhile/add_1:z:00^while/simple_rnn_cell_16/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_16/MatMul/ReadVariableOp1^while/simple_rnn_cell_16/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations0^while/simple_rnn_cell_16/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_16/MatMul/ReadVariableOp1^while/simple_rnn_cell_16/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:00^while/simple_rnn_cell_16/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_16/MatMul/ReadVariableOp1^while/simple_rnn_cell_16/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:00^while/simple_rnn_cell_16/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_16/MatMul/ReadVariableOp1^while/simple_rnn_cell_16/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity!while/simple_rnn_cell_16/Tanh:y:00^while/simple_rnn_cell_16/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_16/MatMul/ReadVariableOp1^while/simple_rnn_cell_16/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2
while/Identity_4")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"v
8while_simple_rnn_cell_16_biasadd_readvariableop_resource:while_simple_rnn_cell_16_biasadd_readvariableop_resource_0"x
9while_simple_rnn_cell_16_matmul_1_readvariableop_resource;while_simple_rnn_cell_16_matmul_1_readvariableop_resource_0"t
7while_simple_rnn_cell_16_matmul_readvariableop_resource9while_simple_rnn_cell_16_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*?
_input_shapes.
,: : : : :??????????: : :::2b
/while/simple_rnn_cell_16/BiasAdd/ReadVariableOp/while/simple_rnn_cell_16/BiasAdd/ReadVariableOp2`
.while/simple_rnn_cell_16/MatMul/ReadVariableOp.while/simple_rnn_cell_16/MatMul/ReadVariableOp2d
0while/simple_rnn_cell_16/MatMul_1/ReadVariableOp0while/simple_rnn_cell_16/MatMul_1/ReadVariableOp: 
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
: :.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?
a
E__inference_dropout_8_layer_call_and_return_conditional_losses_194768

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout/Constx
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:?????????x?2
dropout/Mula
dropout/rateConst*
_output_shapes
: *
dtype0*
valueB
 *???=2
dropout/rate^
dropout/seedConst*
_output_shapes
: *
dtype0*
value	B : 2
dropout/seed?
dropout
MLCDropoutinputsdropout/rate:output:0dropout/seed:output:0*
T0*,
_output_shapes
:?????????x?2	
dropouti
IdentityIdentitydropout:output:0*
T0*,
_output_shapes
:?????????x?2

Identity"
identityIdentity:output:0*+
_input_shapes
:?????????x?:T P
,
_output_shapes
:?????????x?
 
_user_specified_nameinputs
?
?
while_cond_196703
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_196703___redundant_placeholder04
0while_while_cond_196703___redundant_placeholder14
0while_while_cond_196703___redundant_placeholder24
0while_while_cond_196703___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
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
while_identitywhile/Identity:output:0*@
_input_shapes/
-: : : : :?????????2: ::::: 
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
:?????????2:

_output_shapes
: :

_output_shapes
:
?H
?
H__inference_simple_rnn_8_layer_call_and_return_conditional_losses_195886
inputs_05
1simple_rnn_cell_16_matmul_readvariableop_resource6
2simple_rnn_cell_16_biasadd_readvariableop_resource7
3simple_rnn_cell_16_matmul_1_readvariableop_resource
identity??)simple_rnn_cell_16/BiasAdd/ReadVariableOp?(simple_rnn_cell_16/MatMul/ReadVariableOp?*simple_rnn_cell_16/MatMul_1/ReadVariableOp?whileF
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
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
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

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
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
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
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
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
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
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_2?
(simple_rnn_cell_16/MatMul/ReadVariableOpReadVariableOp1simple_rnn_cell_16_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02*
(simple_rnn_cell_16/MatMul/ReadVariableOp?
simple_rnn_cell_16/MatMul	MLCMatMulstrided_slice_2:output:00simple_rnn_cell_16/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
simple_rnn_cell_16/MatMul?
)simple_rnn_cell_16/BiasAdd/ReadVariableOpReadVariableOp2simple_rnn_cell_16_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)simple_rnn_cell_16/BiasAdd/ReadVariableOp?
simple_rnn_cell_16/BiasAddBiasAdd#simple_rnn_cell_16/MatMul:product:01simple_rnn_cell_16/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
simple_rnn_cell_16/BiasAdd?
*simple_rnn_cell_16/MatMul_1/ReadVariableOpReadVariableOp3simple_rnn_cell_16_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02,
*simple_rnn_cell_16/MatMul_1/ReadVariableOp?
simple_rnn_cell_16/MatMul_1	MLCMatMulzeros:output:02simple_rnn_cell_16/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
simple_rnn_cell_16/MatMul_1?
simple_rnn_cell_16/addAddV2#simple_rnn_cell_16/BiasAdd:output:0%simple_rnn_cell_16/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
simple_rnn_cell_16/add?
simple_rnn_cell_16/TanhTanhsimple_rnn_cell_16/add:z:0*
T0*(
_output_shapes
:??????????2
simple_rnn_cell_16/Tanh?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:01simple_rnn_cell_16_matmul_readvariableop_resource2simple_rnn_cell_16_biasadd_readvariableop_resource3simple_rnn_cell_16_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :??????????: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_195820*
condR
while_cond_195819*9
output_shapes(
&: : : : :??????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:???????????????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:???????????????????2
transpose_1?
IdentityIdentitytranspose_1:y:0*^simple_rnn_cell_16/BiasAdd/ReadVariableOp)^simple_rnn_cell_16/MatMul/ReadVariableOp+^simple_rnn_cell_16/MatMul_1/ReadVariableOp^while*
T0*5
_output_shapes#
!:???????????????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????:::2V
)simple_rnn_cell_16/BiasAdd/ReadVariableOp)simple_rnn_cell_16/BiasAdd/ReadVariableOp2T
(simple_rnn_cell_16/MatMul/ReadVariableOp(simple_rnn_cell_16/MatMul/ReadVariableOp2X
*simple_rnn_cell_16/MatMul_1/ReadVariableOp*simple_rnn_cell_16/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :??????????????????
"
_user_specified_name
inputs/0
?
?
-__inference_simple_rnn_9_layer_call_fn_196792
inputs_0
unknown
	unknown_0
	unknown_1
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_simple_rnn_9_layer_call_and_return_conditional_losses_1944802
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*@
_input_shapes/
-:???????????????????:::22
StatefulPartitionedCallStatefulPartitionedCall:_ [
5
_output_shapes#
!:???????????????????
"
_user_specified_name
inputs/0
?
?
while_cond_196065
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_196065___redundant_placeholder04
0while_while_cond_196065___redundant_placeholder14
0while_while_cond_196065___redundant_placeholder24
0while_while_cond_196065___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
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
while_identitywhile/Identity:output:0*A
_input_shapes0
.: : : : :??????????: ::::: 
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
: :.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?
?
N__inference_simple_rnn_cell_17_layer_call_and_return_conditional_losses_194026

inputs

states"
matmul_readvariableop_resource#
biasadd_readvariableop_resource$
 matmul_1_readvariableop_resource
identity

identity_1??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?MatMul_1/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?2*
dtype02
MatMul/ReadVariableOpv
MatMul	MLCMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:2*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22	
BiasAdd?
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:22*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1	MLCMatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22

MatMul_1k
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*'
_output_shapes
:?????????22
addO
TanhTanhadd:z:0*
T0*'
_output_shapes
:?????????22
Tanh?
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????22

Identity?

Identity_1IdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????22

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*F
_input_shapes5
3:??????????:?????????2:::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????2
 
_user_specified_namestates
?H
?
H__inference_simple_rnn_8_layer_call_and_return_conditional_losses_196132

inputs5
1simple_rnn_cell_16_matmul_readvariableop_resource6
2simple_rnn_cell_16_biasadd_readvariableop_resource7
3simple_rnn_cell_16_matmul_1_readvariableop_resource
identity??)simple_rnn_cell_16/BiasAdd/ReadVariableOp?(simple_rnn_cell_16/MatMul/ReadVariableOp?*simple_rnn_cell_16/MatMul_1/ReadVariableOp?whileD
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
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
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

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
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
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:x?????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
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
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
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
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_2?
(simple_rnn_cell_16/MatMul/ReadVariableOpReadVariableOp1simple_rnn_cell_16_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02*
(simple_rnn_cell_16/MatMul/ReadVariableOp?
simple_rnn_cell_16/MatMul	MLCMatMulstrided_slice_2:output:00simple_rnn_cell_16/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
simple_rnn_cell_16/MatMul?
)simple_rnn_cell_16/BiasAdd/ReadVariableOpReadVariableOp2simple_rnn_cell_16_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)simple_rnn_cell_16/BiasAdd/ReadVariableOp?
simple_rnn_cell_16/BiasAddBiasAdd#simple_rnn_cell_16/MatMul:product:01simple_rnn_cell_16/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
simple_rnn_cell_16/BiasAdd?
*simple_rnn_cell_16/MatMul_1/ReadVariableOpReadVariableOp3simple_rnn_cell_16_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02,
*simple_rnn_cell_16/MatMul_1/ReadVariableOp?
simple_rnn_cell_16/MatMul_1	MLCMatMulzeros:output:02simple_rnn_cell_16/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
simple_rnn_cell_16/MatMul_1?
simple_rnn_cell_16/addAddV2#simple_rnn_cell_16/BiasAdd:output:0%simple_rnn_cell_16/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
simple_rnn_cell_16/add?
simple_rnn_cell_16/TanhTanhsimple_rnn_cell_16/add:z:0*
T0*(
_output_shapes
:??????????2
simple_rnn_cell_16/Tanh?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:01simple_rnn_cell_16_matmul_readvariableop_resource2simple_rnn_cell_16_biasadd_readvariableop_resource3simple_rnn_cell_16_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :??????????: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_196066*
condR
while_cond_196065*9
output_shapes(
&: : : : :??????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:x??????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:?????????x?2
transpose_1?
IdentityIdentitytranspose_1:y:0*^simple_rnn_cell_16/BiasAdd/ReadVariableOp)^simple_rnn_cell_16/MatMul/ReadVariableOp+^simple_rnn_cell_16/MatMul_1/ReadVariableOp^while*
T0*,
_output_shapes
:?????????x?2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????x:::2V
)simple_rnn_cell_16/BiasAdd/ReadVariableOp)simple_rnn_cell_16/BiasAdd/ReadVariableOp2T
(simple_rnn_cell_16/MatMul/ReadVariableOp(simple_rnn_cell_16/MatMul/ReadVariableOp2X
*simple_rnn_cell_16/MatMul_1/ReadVariableOp*simple_rnn_cell_16/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:?????????x
 
_user_specified_nameinputs
?
?
.__inference_sequential_24_layer_call_fn_195183
simple_rnn_8_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallsimple_rnn_8_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????<**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_sequential_24_layer_call_and_return_conditional_losses_1951642
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????<2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:?????????x::::::::22
StatefulPartitionedCallStatefulPartitionedCall:_ [
+
_output_shapes
:?????????x
,
_user_specified_namesimple_rnn_8_input
?
?
N__inference_simple_rnn_cell_16_layer_call_and_return_conditional_losses_196863

inputs
states_0"
matmul_readvariableop_resource#
biasadd_readvariableop_resource$
 matmul_1_readvariableop_resource
identity

identity_1??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?MatMul_1/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOpw
MatMul	MLCMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAdd?
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul_1/ReadVariableOp
MatMul_1	MLCMatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2

MatMul_1l
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
addP
TanhTanhadd:z:0*
T0*(
_output_shapes
:??????????2
Tanh?
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity?

Identity_1IdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*F
_input_shapes5
3:?????????:??????????:::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/0
?	
?
3__inference_simple_rnn_cell_17_layer_call_fn_196970

inputs
states_0
unknown
	unknown_0
	unknown_1
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????2:?????????2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_simple_rnn_cell_17_layer_call_and_return_conditional_losses_1940432
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????22

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????22

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*F
_input_shapes5
3:??????????:?????????2:::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:QM
'
_output_shapes
:?????????2
"
_user_specified_name
states/0
?
F
*__inference_dropout_8_layer_call_fn_196295

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????x?* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_8_layer_call_and_return_conditional_losses_1947682
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:?????????x?2

Identity"
identityIdentity:output:0*+
_input_shapes
:?????????x?:T P
,
_output_shapes
:?????????x?
 
_user_specified_nameinputs
?
?
-__inference_simple_rnn_9_layer_call_fn_196535

inputs
unknown
	unknown_0
	unknown_1
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_simple_rnn_9_layer_call_and_return_conditional_losses_1948972
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????x?:::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:?????????x?
 
_user_specified_nameinputs
?R
?
,sequential_24_simple_rnn_9_while_body_193390R
Nsequential_24_simple_rnn_9_while_sequential_24_simple_rnn_9_while_loop_counterX
Tsequential_24_simple_rnn_9_while_sequential_24_simple_rnn_9_while_maximum_iterations0
,sequential_24_simple_rnn_9_while_placeholder2
.sequential_24_simple_rnn_9_while_placeholder_12
.sequential_24_simple_rnn_9_while_placeholder_2Q
Msequential_24_simple_rnn_9_while_sequential_24_simple_rnn_9_strided_slice_1_0?
?sequential_24_simple_rnn_9_while_tensorarrayv2read_tensorlistgetitem_sequential_24_simple_rnn_9_tensorarrayunstack_tensorlistfromtensor_0X
Tsequential_24_simple_rnn_9_while_simple_rnn_cell_17_matmul_readvariableop_resource_0Y
Usequential_24_simple_rnn_9_while_simple_rnn_cell_17_biasadd_readvariableop_resource_0Z
Vsequential_24_simple_rnn_9_while_simple_rnn_cell_17_matmul_1_readvariableop_resource_0-
)sequential_24_simple_rnn_9_while_identity/
+sequential_24_simple_rnn_9_while_identity_1/
+sequential_24_simple_rnn_9_while_identity_2/
+sequential_24_simple_rnn_9_while_identity_3/
+sequential_24_simple_rnn_9_while_identity_4O
Ksequential_24_simple_rnn_9_while_sequential_24_simple_rnn_9_strided_slice_1?
?sequential_24_simple_rnn_9_while_tensorarrayv2read_tensorlistgetitem_sequential_24_simple_rnn_9_tensorarrayunstack_tensorlistfromtensorV
Rsequential_24_simple_rnn_9_while_simple_rnn_cell_17_matmul_readvariableop_resourceW
Ssequential_24_simple_rnn_9_while_simple_rnn_cell_17_biasadd_readvariableop_resourceX
Tsequential_24_simple_rnn_9_while_simple_rnn_cell_17_matmul_1_readvariableop_resource??Jsequential_24/simple_rnn_9/while/simple_rnn_cell_17/BiasAdd/ReadVariableOp?Isequential_24/simple_rnn_9/while/simple_rnn_cell_17/MatMul/ReadVariableOp?Ksequential_24/simple_rnn_9/while/simple_rnn_cell_17/MatMul_1/ReadVariableOp?
Rsequential_24/simple_rnn_9/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2T
Rsequential_24/simple_rnn_9/while/TensorArrayV2Read/TensorListGetItem/element_shape?
Dsequential_24/simple_rnn_9/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem?sequential_24_simple_rnn_9_while_tensorarrayv2read_tensorlistgetitem_sequential_24_simple_rnn_9_tensorarrayunstack_tensorlistfromtensor_0,sequential_24_simple_rnn_9_while_placeholder[sequential_24/simple_rnn_9/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype02F
Dsequential_24/simple_rnn_9/while/TensorArrayV2Read/TensorListGetItem?
Isequential_24/simple_rnn_9/while/simple_rnn_cell_17/MatMul/ReadVariableOpReadVariableOpTsequential_24_simple_rnn_9_while_simple_rnn_cell_17_matmul_readvariableop_resource_0*
_output_shapes
:	?2*
dtype02K
Isequential_24/simple_rnn_9/while/simple_rnn_cell_17/MatMul/ReadVariableOp?
:sequential_24/simple_rnn_9/while/simple_rnn_cell_17/MatMul	MLCMatMulKsequential_24/simple_rnn_9/while/TensorArrayV2Read/TensorListGetItem:item:0Qsequential_24/simple_rnn_9/while/simple_rnn_cell_17/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22<
:sequential_24/simple_rnn_9/while/simple_rnn_cell_17/MatMul?
Jsequential_24/simple_rnn_9/while/simple_rnn_cell_17/BiasAdd/ReadVariableOpReadVariableOpUsequential_24_simple_rnn_9_while_simple_rnn_cell_17_biasadd_readvariableop_resource_0*
_output_shapes
:2*
dtype02L
Jsequential_24/simple_rnn_9/while/simple_rnn_cell_17/BiasAdd/ReadVariableOp?
;sequential_24/simple_rnn_9/while/simple_rnn_cell_17/BiasAddBiasAddDsequential_24/simple_rnn_9/while/simple_rnn_cell_17/MatMul:product:0Rsequential_24/simple_rnn_9/while/simple_rnn_cell_17/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22=
;sequential_24/simple_rnn_9/while/simple_rnn_cell_17/BiasAdd?
Ksequential_24/simple_rnn_9/while/simple_rnn_cell_17/MatMul_1/ReadVariableOpReadVariableOpVsequential_24_simple_rnn_9_while_simple_rnn_cell_17_matmul_1_readvariableop_resource_0*
_output_shapes

:22*
dtype02M
Ksequential_24/simple_rnn_9/while/simple_rnn_cell_17/MatMul_1/ReadVariableOp?
<sequential_24/simple_rnn_9/while/simple_rnn_cell_17/MatMul_1	MLCMatMul.sequential_24_simple_rnn_9_while_placeholder_2Ssequential_24/simple_rnn_9/while/simple_rnn_cell_17/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22>
<sequential_24/simple_rnn_9/while/simple_rnn_cell_17/MatMul_1?
7sequential_24/simple_rnn_9/while/simple_rnn_cell_17/addAddV2Dsequential_24/simple_rnn_9/while/simple_rnn_cell_17/BiasAdd:output:0Fsequential_24/simple_rnn_9/while/simple_rnn_cell_17/MatMul_1:product:0*
T0*'
_output_shapes
:?????????229
7sequential_24/simple_rnn_9/while/simple_rnn_cell_17/add?
8sequential_24/simple_rnn_9/while/simple_rnn_cell_17/TanhTanh;sequential_24/simple_rnn_9/while/simple_rnn_cell_17/add:z:0*
T0*'
_output_shapes
:?????????22:
8sequential_24/simple_rnn_9/while/simple_rnn_cell_17/Tanh?
Esequential_24/simple_rnn_9/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem.sequential_24_simple_rnn_9_while_placeholder_1,sequential_24_simple_rnn_9_while_placeholder<sequential_24/simple_rnn_9/while/simple_rnn_cell_17/Tanh:y:0*
_output_shapes
: *
element_dtype02G
Esequential_24/simple_rnn_9/while/TensorArrayV2Write/TensorListSetItem?
&sequential_24/simple_rnn_9/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2(
&sequential_24/simple_rnn_9/while/add/y?
$sequential_24/simple_rnn_9/while/addAddV2,sequential_24_simple_rnn_9_while_placeholder/sequential_24/simple_rnn_9/while/add/y:output:0*
T0*
_output_shapes
: 2&
$sequential_24/simple_rnn_9/while/add?
(sequential_24/simple_rnn_9/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2*
(sequential_24/simple_rnn_9/while/add_1/y?
&sequential_24/simple_rnn_9/while/add_1AddV2Nsequential_24_simple_rnn_9_while_sequential_24_simple_rnn_9_while_loop_counter1sequential_24/simple_rnn_9/while/add_1/y:output:0*
T0*
_output_shapes
: 2(
&sequential_24/simple_rnn_9/while/add_1?
)sequential_24/simple_rnn_9/while/IdentityIdentity*sequential_24/simple_rnn_9/while/add_1:z:0K^sequential_24/simple_rnn_9/while/simple_rnn_cell_17/BiasAdd/ReadVariableOpJ^sequential_24/simple_rnn_9/while/simple_rnn_cell_17/MatMul/ReadVariableOpL^sequential_24/simple_rnn_9/while/simple_rnn_cell_17/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2+
)sequential_24/simple_rnn_9/while/Identity?
+sequential_24/simple_rnn_9/while/Identity_1IdentityTsequential_24_simple_rnn_9_while_sequential_24_simple_rnn_9_while_maximum_iterationsK^sequential_24/simple_rnn_9/while/simple_rnn_cell_17/BiasAdd/ReadVariableOpJ^sequential_24/simple_rnn_9/while/simple_rnn_cell_17/MatMul/ReadVariableOpL^sequential_24/simple_rnn_9/while/simple_rnn_cell_17/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2-
+sequential_24/simple_rnn_9/while/Identity_1?
+sequential_24/simple_rnn_9/while/Identity_2Identity(sequential_24/simple_rnn_9/while/add:z:0K^sequential_24/simple_rnn_9/while/simple_rnn_cell_17/BiasAdd/ReadVariableOpJ^sequential_24/simple_rnn_9/while/simple_rnn_cell_17/MatMul/ReadVariableOpL^sequential_24/simple_rnn_9/while/simple_rnn_cell_17/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2-
+sequential_24/simple_rnn_9/while/Identity_2?
+sequential_24/simple_rnn_9/while/Identity_3IdentityUsequential_24/simple_rnn_9/while/TensorArrayV2Write/TensorListSetItem:output_handle:0K^sequential_24/simple_rnn_9/while/simple_rnn_cell_17/BiasAdd/ReadVariableOpJ^sequential_24/simple_rnn_9/while/simple_rnn_cell_17/MatMul/ReadVariableOpL^sequential_24/simple_rnn_9/while/simple_rnn_cell_17/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2-
+sequential_24/simple_rnn_9/while/Identity_3?
+sequential_24/simple_rnn_9/while/Identity_4Identity<sequential_24/simple_rnn_9/while/simple_rnn_cell_17/Tanh:y:0K^sequential_24/simple_rnn_9/while/simple_rnn_cell_17/BiasAdd/ReadVariableOpJ^sequential_24/simple_rnn_9/while/simple_rnn_cell_17/MatMul/ReadVariableOpL^sequential_24/simple_rnn_9/while/simple_rnn_cell_17/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????22-
+sequential_24/simple_rnn_9/while/Identity_4"_
)sequential_24_simple_rnn_9_while_identity2sequential_24/simple_rnn_9/while/Identity:output:0"c
+sequential_24_simple_rnn_9_while_identity_14sequential_24/simple_rnn_9/while/Identity_1:output:0"c
+sequential_24_simple_rnn_9_while_identity_24sequential_24/simple_rnn_9/while/Identity_2:output:0"c
+sequential_24_simple_rnn_9_while_identity_34sequential_24/simple_rnn_9/while/Identity_3:output:0"c
+sequential_24_simple_rnn_9_while_identity_44sequential_24/simple_rnn_9/while/Identity_4:output:0"?
Ksequential_24_simple_rnn_9_while_sequential_24_simple_rnn_9_strided_slice_1Msequential_24_simple_rnn_9_while_sequential_24_simple_rnn_9_strided_slice_1_0"?
Ssequential_24_simple_rnn_9_while_simple_rnn_cell_17_biasadd_readvariableop_resourceUsequential_24_simple_rnn_9_while_simple_rnn_cell_17_biasadd_readvariableop_resource_0"?
Tsequential_24_simple_rnn_9_while_simple_rnn_cell_17_matmul_1_readvariableop_resourceVsequential_24_simple_rnn_9_while_simple_rnn_cell_17_matmul_1_readvariableop_resource_0"?
Rsequential_24_simple_rnn_9_while_simple_rnn_cell_17_matmul_readvariableop_resourceTsequential_24_simple_rnn_9_while_simple_rnn_cell_17_matmul_readvariableop_resource_0"?
?sequential_24_simple_rnn_9_while_tensorarrayv2read_tensorlistgetitem_sequential_24_simple_rnn_9_tensorarrayunstack_tensorlistfromtensor?sequential_24_simple_rnn_9_while_tensorarrayv2read_tensorlistgetitem_sequential_24_simple_rnn_9_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :?????????2: : :::2?
Jsequential_24/simple_rnn_9/while/simple_rnn_cell_17/BiasAdd/ReadVariableOpJsequential_24/simple_rnn_9/while/simple_rnn_cell_17/BiasAdd/ReadVariableOp2?
Isequential_24/simple_rnn_9/while/simple_rnn_cell_17/MatMul/ReadVariableOpIsequential_24/simple_rnn_9/while/simple_rnn_cell_17/MatMul/ReadVariableOp2?
Ksequential_24/simple_rnn_9/while/simple_rnn_cell_17/MatMul_1/ReadVariableOpKsequential_24/simple_rnn_9/while/simple_rnn_cell_17/MatMul_1/ReadVariableOp: 
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
:?????????2:

_output_shapes
: :

_output_shapes
: 
?H
?
H__inference_simple_rnn_8_layer_call_and_return_conditional_losses_194604

inputs5
1simple_rnn_cell_16_matmul_readvariableop_resource6
2simple_rnn_cell_16_biasadd_readvariableop_resource7
3simple_rnn_cell_16_matmul_1_readvariableop_resource
identity??)simple_rnn_cell_16/BiasAdd/ReadVariableOp?(simple_rnn_cell_16/MatMul/ReadVariableOp?*simple_rnn_cell_16/MatMul_1/ReadVariableOp?whileD
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
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
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

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
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
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:x?????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
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
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
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
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_2?
(simple_rnn_cell_16/MatMul/ReadVariableOpReadVariableOp1simple_rnn_cell_16_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02*
(simple_rnn_cell_16/MatMul/ReadVariableOp?
simple_rnn_cell_16/MatMul	MLCMatMulstrided_slice_2:output:00simple_rnn_cell_16/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
simple_rnn_cell_16/MatMul?
)simple_rnn_cell_16/BiasAdd/ReadVariableOpReadVariableOp2simple_rnn_cell_16_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)simple_rnn_cell_16/BiasAdd/ReadVariableOp?
simple_rnn_cell_16/BiasAddBiasAdd#simple_rnn_cell_16/MatMul:product:01simple_rnn_cell_16/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
simple_rnn_cell_16/BiasAdd?
*simple_rnn_cell_16/MatMul_1/ReadVariableOpReadVariableOp3simple_rnn_cell_16_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02,
*simple_rnn_cell_16/MatMul_1/ReadVariableOp?
simple_rnn_cell_16/MatMul_1	MLCMatMulzeros:output:02simple_rnn_cell_16/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
simple_rnn_cell_16/MatMul_1?
simple_rnn_cell_16/addAddV2#simple_rnn_cell_16/BiasAdd:output:0%simple_rnn_cell_16/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
simple_rnn_cell_16/add?
simple_rnn_cell_16/TanhTanhsimple_rnn_cell_16/add:z:0*
T0*(
_output_shapes
:??????????2
simple_rnn_cell_16/Tanh?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:01simple_rnn_cell_16_matmul_readvariableop_resource2simple_rnn_cell_16_biasadd_readvariableop_resource3simple_rnn_cell_16_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :??????????: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_194538*
condR
while_cond_194537*9
output_shapes(
&: : : : :??????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:x??????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:?????????x?2
transpose_1?
IdentityIdentitytranspose_1:y:0*^simple_rnn_cell_16/BiasAdd/ReadVariableOp)^simple_rnn_cell_16/MatMul/ReadVariableOp+^simple_rnn_cell_16/MatMul_1/ReadVariableOp^while*
T0*,
_output_shapes
:?????????x?2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????x:::2V
)simple_rnn_cell_16/BiasAdd/ReadVariableOp)simple_rnn_cell_16/BiasAdd/ReadVariableOp2T
(simple_rnn_cell_16/MatMul/ReadVariableOp(simple_rnn_cell_16/MatMul/ReadVariableOp2X
*simple_rnn_cell_16/MatMul_1/ReadVariableOp*simple_rnn_cell_16/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:?????????x
 
_user_specified_nameinputs
?
?
I__inference_sequential_24_layer_call_and_return_conditional_losses_195164

inputs
simple_rnn_8_195140
simple_rnn_8_195142
simple_rnn_8_195144
simple_rnn_9_195149
simple_rnn_9_195151
simple_rnn_9_195153
dense_28_195158
dense_28_195160
identity?? dense_28/StatefulPartitionedCall?$simple_rnn_8/StatefulPartitionedCall?$simple_rnn_9/StatefulPartitionedCall?
$simple_rnn_8/StatefulPartitionedCallStatefulPartitionedCallinputssimple_rnn_8_195140simple_rnn_8_195142simple_rnn_8_195144*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????x?*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_simple_rnn_8_layer_call_and_return_conditional_losses_1946042&
$simple_rnn_8/StatefulPartitionedCall?
activation_20/PartitionedCallPartitionedCall-simple_rnn_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????x?* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_activation_20_layer_call_and_return_conditional_losses_1947512
activation_20/PartitionedCall?
dropout_8/PartitionedCallPartitionedCall&activation_20/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????x?* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_8_layer_call_and_return_conditional_losses_1947682
dropout_8/PartitionedCall?
$simple_rnn_9/StatefulPartitionedCallStatefulPartitionedCall"dropout_8/PartitionedCall:output:0simple_rnn_9_195149simple_rnn_9_195151simple_rnn_9_195153*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_simple_rnn_9_layer_call_and_return_conditional_losses_1948972&
$simple_rnn_9/StatefulPartitionedCall?
activation_21/PartitionedCallPartitionedCall-simple_rnn_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_activation_21_layer_call_and_return_conditional_losses_1950442
activation_21/PartitionedCall?
dropout_9/PartitionedCallPartitionedCall&activation_21/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_9_layer_call_and_return_conditional_losses_1950612
dropout_9/PartitionedCall?
 dense_28/StatefulPartitionedCallStatefulPartitionedCall"dropout_9/PartitionedCall:output:0dense_28_195158dense_28_195160*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????<*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_28_layer_call_and_return_conditional_losses_1950902"
 dense_28/StatefulPartitionedCall?
IdentityIdentity)dense_28/StatefulPartitionedCall:output:0!^dense_28/StatefulPartitionedCall%^simple_rnn_8/StatefulPartitionedCall%^simple_rnn_9/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????<2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:?????????x::::::::2D
 dense_28/StatefulPartitionedCall dense_28/StatefulPartitionedCall2L
$simple_rnn_8/StatefulPartitionedCall$simple_rnn_8/StatefulPartitionedCall2L
$simple_rnn_9/StatefulPartitionedCall$simple_rnn_9/StatefulPartitionedCall:S O
+
_output_shapes
:?????????x
 
_user_specified_nameinputs
?
?
,sequential_24_simple_rnn_8_while_cond_193279R
Nsequential_24_simple_rnn_8_while_sequential_24_simple_rnn_8_while_loop_counterX
Tsequential_24_simple_rnn_8_while_sequential_24_simple_rnn_8_while_maximum_iterations0
,sequential_24_simple_rnn_8_while_placeholder2
.sequential_24_simple_rnn_8_while_placeholder_12
.sequential_24_simple_rnn_8_while_placeholder_2T
Psequential_24_simple_rnn_8_while_less_sequential_24_simple_rnn_8_strided_slice_1j
fsequential_24_simple_rnn_8_while_sequential_24_simple_rnn_8_while_cond_193279___redundant_placeholder0j
fsequential_24_simple_rnn_8_while_sequential_24_simple_rnn_8_while_cond_193279___redundant_placeholder1j
fsequential_24_simple_rnn_8_while_sequential_24_simple_rnn_8_while_cond_193279___redundant_placeholder2j
fsequential_24_simple_rnn_8_while_sequential_24_simple_rnn_8_while_cond_193279___redundant_placeholder3-
)sequential_24_simple_rnn_8_while_identity
?
%sequential_24/simple_rnn_8/while/LessLess,sequential_24_simple_rnn_8_while_placeholderPsequential_24_simple_rnn_8_while_less_sequential_24_simple_rnn_8_strided_slice_1*
T0*
_output_shapes
: 2'
%sequential_24/simple_rnn_8/while/Less?
)sequential_24/simple_rnn_8/while/IdentityIdentity)sequential_24/simple_rnn_8/while/Less:z:0*
T0
*
_output_shapes
: 2+
)sequential_24/simple_rnn_8/while/Identity"_
)sequential_24_simple_rnn_8_while_identity2sequential_24/simple_rnn_8/while/Identity:output:0*A
_input_shapes0
.: : : : :??????????: ::::: 
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
: :.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?
?
while_cond_196177
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_196177___redundant_placeholder04
0while_while_cond_196177___redundant_placeholder14
0while_while_cond_196177___redundant_placeholder24
0while_while_cond_196177___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
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
while_identitywhile/Identity:output:0*A
_input_shapes0
.: : : : :??????????: ::::: 
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
: :.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?

?
simple_rnn_8_while_cond_1953076
2simple_rnn_8_while_simple_rnn_8_while_loop_counter<
8simple_rnn_8_while_simple_rnn_8_while_maximum_iterations"
simple_rnn_8_while_placeholder$
 simple_rnn_8_while_placeholder_1$
 simple_rnn_8_while_placeholder_28
4simple_rnn_8_while_less_simple_rnn_8_strided_slice_1N
Jsimple_rnn_8_while_simple_rnn_8_while_cond_195307___redundant_placeholder0N
Jsimple_rnn_8_while_simple_rnn_8_while_cond_195307___redundant_placeholder1N
Jsimple_rnn_8_while_simple_rnn_8_while_cond_195307___redundant_placeholder2N
Jsimple_rnn_8_while_simple_rnn_8_while_cond_195307___redundant_placeholder3
simple_rnn_8_while_identity
?
simple_rnn_8/while/LessLesssimple_rnn_8_while_placeholder4simple_rnn_8_while_less_simple_rnn_8_strided_slice_1*
T0*
_output_shapes
: 2
simple_rnn_8/while/Less?
simple_rnn_8/while/IdentityIdentitysimple_rnn_8/while/Less:z:0*
T0
*
_output_shapes
: 2
simple_rnn_8/while/Identity"C
simple_rnn_8_while_identity$simple_rnn_8/while/Identity:output:0*A
_input_shapes0
.: : : : :??????????: ::::: 
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
: :.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?	
?
3__inference_simple_rnn_cell_17_layer_call_fn_196956

inputs
states_0
unknown
	unknown_0
	unknown_1
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????2:?????????2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_simple_rnn_cell_17_layer_call_and_return_conditional_losses_1940262
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????22

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????22

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*F
_input_shapes5
3:??????????:?????????2:::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:QM
'
_output_shapes
:?????????2
"
_user_specified_name
states/0
?
c
E__inference_dropout_9_layer_call_and_return_conditional_losses_195066

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????22

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:?????????22

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:?????????2:O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs
?<
?
H__inference_simple_rnn_9_layer_call_and_return_conditional_losses_194363

inputs
simple_rnn_cell_17_194288
simple_rnn_cell_17_194290
simple_rnn_cell_17_194292
identity??*simple_rnn_cell_17/StatefulPartitionedCall?whileD
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
value	B :22
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
value	B :22
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
:?????????22
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:???????????????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
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
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_2?
*simple_rnn_cell_17/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0simple_rnn_cell_17_194288simple_rnn_cell_17_194290simple_rnn_cell_17_194292*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????2:?????????2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_simple_rnn_cell_17_layer_call_and_return_conditional_losses_1940262,
*simple_rnn_cell_17/StatefulPartitionedCall?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
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
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0simple_rnn_cell_17_194288simple_rnn_cell_17_194290simple_rnn_cell_17_194292*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????2: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_194300*
condR
while_cond_194299*8
output_shapes'
%: : : : :?????????2: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????2*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????2*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????22
transpose_1?
IdentityIdentitystrided_slice_3:output:0+^simple_rnn_cell_17/StatefulPartitionedCall^while*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*@
_input_shapes/
-:???????????????????:::2X
*simple_rnn_cell_17/StatefulPartitionedCall*simple_rnn_cell_17/StatefulPartitionedCall2
whilewhile:] Y
5
_output_shapes#
!:???????????????????
 
_user_specified_nameinputs
?
?
N__inference_simple_rnn_cell_16_layer_call_and_return_conditional_losses_196880

inputs
states_0"
matmul_readvariableop_resource#
biasadd_readvariableop_resource$
 matmul_1_readvariableop_resource
identity

identity_1??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?MatMul_1/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOpw
MatMul	MLCMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAdd?
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul_1/ReadVariableOp
MatMul_1	MLCMatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2

MatMul_1l
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
addP
TanhTanhadd:z:0*
T0*(
_output_shapes
:??????????2
Tanh?
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity?

Identity_1IdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*F
_input_shapes5
3:?????????:??????????:::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/0
?H
?
H__inference_simple_rnn_9_layer_call_and_return_conditional_losses_196524

inputs5
1simple_rnn_cell_17_matmul_readvariableop_resource6
2simple_rnn_cell_17_biasadd_readvariableop_resource7
3simple_rnn_cell_17_matmul_1_readvariableop_resource
identity??)simple_rnn_cell_17/BiasAdd/ReadVariableOp?(simple_rnn_cell_17/MatMul/ReadVariableOp?*simple_rnn_cell_17/MatMul_1/ReadVariableOp?whileD
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
value	B :22
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
value	B :22
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
:?????????22
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm{
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:x??????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
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
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_2?
(simple_rnn_cell_17/MatMul/ReadVariableOpReadVariableOp1simple_rnn_cell_17_matmul_readvariableop_resource*
_output_shapes
:	?2*
dtype02*
(simple_rnn_cell_17/MatMul/ReadVariableOp?
simple_rnn_cell_17/MatMul	MLCMatMulstrided_slice_2:output:00simple_rnn_cell_17/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
simple_rnn_cell_17/MatMul?
)simple_rnn_cell_17/BiasAdd/ReadVariableOpReadVariableOp2simple_rnn_cell_17_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02+
)simple_rnn_cell_17/BiasAdd/ReadVariableOp?
simple_rnn_cell_17/BiasAddBiasAdd#simple_rnn_cell_17/MatMul:product:01simple_rnn_cell_17/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
simple_rnn_cell_17/BiasAdd?
*simple_rnn_cell_17/MatMul_1/ReadVariableOpReadVariableOp3simple_rnn_cell_17_matmul_1_readvariableop_resource*
_output_shapes

:22*
dtype02,
*simple_rnn_cell_17/MatMul_1/ReadVariableOp?
simple_rnn_cell_17/MatMul_1	MLCMatMulzeros:output:02simple_rnn_cell_17/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
simple_rnn_cell_17/MatMul_1?
simple_rnn_cell_17/addAddV2#simple_rnn_cell_17/BiasAdd:output:0%simple_rnn_cell_17/MatMul_1:product:0*
T0*'
_output_shapes
:?????????22
simple_rnn_cell_17/add?
simple_rnn_cell_17/TanhTanhsimple_rnn_cell_17/add:z:0*
T0*'
_output_shapes
:?????????22
simple_rnn_cell_17/Tanh?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:01simple_rnn_cell_17_matmul_readvariableop_resource2simple_rnn_cell_17_biasadd_readvariableop_resource3simple_rnn_cell_17_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????2: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_196458*
condR
while_cond_196457*8
output_shapes'
%: : : : :?????????2: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:x?????????2*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????2*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????x22
transpose_1?
IdentityIdentitystrided_slice_3:output:0*^simple_rnn_cell_17/BiasAdd/ReadVariableOp)^simple_rnn_cell_17/MatMul/ReadVariableOp+^simple_rnn_cell_17/MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????x?:::2V
)simple_rnn_cell_17/BiasAdd/ReadVariableOp)simple_rnn_cell_17/BiasAdd/ReadVariableOp2T
(simple_rnn_cell_17/MatMul/ReadVariableOp(simple_rnn_cell_17/MatMul/ReadVariableOp2X
*simple_rnn_cell_17/MatMul_1/ReadVariableOp*simple_rnn_cell_17/MatMul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:?????????x?
 
_user_specified_nameinputs
?
a
E__inference_dropout_9_layer_call_and_return_conditional_losses_195061

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:?????????22
dropout/Mula
dropout/rateConst*
_output_shapes
: *
dtype0*
valueB
 *???=2
dropout/rate^
dropout/seedConst*
_output_shapes
: *
dtype0*
value	B : 2
dropout/seed?
dropout
MLCDropoutinputsdropout/rate:output:0dropout/seed:output:0*
T0*'
_output_shapes
:?????????22	
dropoutd
IdentityIdentitydropout:output:0*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????2:O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs
??
?
I__inference_sequential_24_layer_call_and_return_conditional_losses_195501

inputsB
>simple_rnn_8_simple_rnn_cell_16_matmul_readvariableop_resourceC
?simple_rnn_8_simple_rnn_cell_16_biasadd_readvariableop_resourceD
@simple_rnn_8_simple_rnn_cell_16_matmul_1_readvariableop_resourceB
>simple_rnn_9_simple_rnn_cell_17_matmul_readvariableop_resourceC
?simple_rnn_9_simple_rnn_cell_17_biasadd_readvariableop_resourceD
@simple_rnn_9_simple_rnn_cell_17_matmul_1_readvariableop_resource.
*dense_28_mlcmatmul_readvariableop_resource,
(dense_28_biasadd_readvariableop_resource
identity??dense_28/BiasAdd/ReadVariableOp?!dense_28/MLCMatMul/ReadVariableOp?6simple_rnn_8/simple_rnn_cell_16/BiasAdd/ReadVariableOp?5simple_rnn_8/simple_rnn_cell_16/MatMul/ReadVariableOp?7simple_rnn_8/simple_rnn_cell_16/MatMul_1/ReadVariableOp?simple_rnn_8/while?6simple_rnn_9/simple_rnn_cell_17/BiasAdd/ReadVariableOp?5simple_rnn_9/simple_rnn_cell_17/MatMul/ReadVariableOp?7simple_rnn_9/simple_rnn_cell_17/MatMul_1/ReadVariableOp?simple_rnn_9/while^
simple_rnn_8/ShapeShapeinputs*
T0*
_output_shapes
:2
simple_rnn_8/Shape?
 simple_rnn_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 simple_rnn_8/strided_slice/stack?
"simple_rnn_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"simple_rnn_8/strided_slice/stack_1?
"simple_rnn_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"simple_rnn_8/strided_slice/stack_2?
simple_rnn_8/strided_sliceStridedSlicesimple_rnn_8/Shape:output:0)simple_rnn_8/strided_slice/stack:output:0+simple_rnn_8/strided_slice/stack_1:output:0+simple_rnn_8/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
simple_rnn_8/strided_slicew
simple_rnn_8/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
simple_rnn_8/zeros/mul/y?
simple_rnn_8/zeros/mulMul#simple_rnn_8/strided_slice:output:0!simple_rnn_8/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
simple_rnn_8/zeros/muly
simple_rnn_8/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
simple_rnn_8/zeros/Less/y?
simple_rnn_8/zeros/LessLesssimple_rnn_8/zeros/mul:z:0"simple_rnn_8/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
simple_rnn_8/zeros/Less}
simple_rnn_8/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
simple_rnn_8/zeros/packed/1?
simple_rnn_8/zeros/packedPack#simple_rnn_8/strided_slice:output:0$simple_rnn_8/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
simple_rnn_8/zeros/packedy
simple_rnn_8/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
simple_rnn_8/zeros/Const?
simple_rnn_8/zerosFill"simple_rnn_8/zeros/packed:output:0!simple_rnn_8/zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
simple_rnn_8/zeros?
simple_rnn_8/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
simple_rnn_8/transpose/perm?
simple_rnn_8/transpose	Transposeinputs$simple_rnn_8/transpose/perm:output:0*
T0*+
_output_shapes
:x?????????2
simple_rnn_8/transposev
simple_rnn_8/Shape_1Shapesimple_rnn_8/transpose:y:0*
T0*
_output_shapes
:2
simple_rnn_8/Shape_1?
"simple_rnn_8/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"simple_rnn_8/strided_slice_1/stack?
$simple_rnn_8/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2&
$simple_rnn_8/strided_slice_1/stack_1?
$simple_rnn_8/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$simple_rnn_8/strided_slice_1/stack_2?
simple_rnn_8/strided_slice_1StridedSlicesimple_rnn_8/Shape_1:output:0+simple_rnn_8/strided_slice_1/stack:output:0-simple_rnn_8/strided_slice_1/stack_1:output:0-simple_rnn_8/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
simple_rnn_8/strided_slice_1?
(simple_rnn_8/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2*
(simple_rnn_8/TensorArrayV2/element_shape?
simple_rnn_8/TensorArrayV2TensorListReserve1simple_rnn_8/TensorArrayV2/element_shape:output:0%simple_rnn_8/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
simple_rnn_8/TensorArrayV2?
Bsimple_rnn_8/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2D
Bsimple_rnn_8/TensorArrayUnstack/TensorListFromTensor/element_shape?
4simple_rnn_8/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorsimple_rnn_8/transpose:y:0Ksimple_rnn_8/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type026
4simple_rnn_8/TensorArrayUnstack/TensorListFromTensor?
"simple_rnn_8/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"simple_rnn_8/strided_slice_2/stack?
$simple_rnn_8/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2&
$simple_rnn_8/strided_slice_2/stack_1?
$simple_rnn_8/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$simple_rnn_8/strided_slice_2/stack_2?
simple_rnn_8/strided_slice_2StridedSlicesimple_rnn_8/transpose:y:0+simple_rnn_8/strided_slice_2/stack:output:0-simple_rnn_8/strided_slice_2/stack_1:output:0-simple_rnn_8/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
simple_rnn_8/strided_slice_2?
5simple_rnn_8/simple_rnn_cell_16/MatMul/ReadVariableOpReadVariableOp>simple_rnn_8_simple_rnn_cell_16_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype027
5simple_rnn_8/simple_rnn_cell_16/MatMul/ReadVariableOp?
&simple_rnn_8/simple_rnn_cell_16/MatMul	MLCMatMul%simple_rnn_8/strided_slice_2:output:0=simple_rnn_8/simple_rnn_cell_16/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2(
&simple_rnn_8/simple_rnn_cell_16/MatMul?
6simple_rnn_8/simple_rnn_cell_16/BiasAdd/ReadVariableOpReadVariableOp?simple_rnn_8_simple_rnn_cell_16_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype028
6simple_rnn_8/simple_rnn_cell_16/BiasAdd/ReadVariableOp?
'simple_rnn_8/simple_rnn_cell_16/BiasAddBiasAdd0simple_rnn_8/simple_rnn_cell_16/MatMul:product:0>simple_rnn_8/simple_rnn_cell_16/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2)
'simple_rnn_8/simple_rnn_cell_16/BiasAdd?
7simple_rnn_8/simple_rnn_cell_16/MatMul_1/ReadVariableOpReadVariableOp@simple_rnn_8_simple_rnn_cell_16_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype029
7simple_rnn_8/simple_rnn_cell_16/MatMul_1/ReadVariableOp?
(simple_rnn_8/simple_rnn_cell_16/MatMul_1	MLCMatMulsimple_rnn_8/zeros:output:0?simple_rnn_8/simple_rnn_cell_16/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2*
(simple_rnn_8/simple_rnn_cell_16/MatMul_1?
#simple_rnn_8/simple_rnn_cell_16/addAddV20simple_rnn_8/simple_rnn_cell_16/BiasAdd:output:02simple_rnn_8/simple_rnn_cell_16/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2%
#simple_rnn_8/simple_rnn_cell_16/add?
$simple_rnn_8/simple_rnn_cell_16/TanhTanh'simple_rnn_8/simple_rnn_cell_16/add:z:0*
T0*(
_output_shapes
:??????????2&
$simple_rnn_8/simple_rnn_cell_16/Tanh?
*simple_rnn_8/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2,
*simple_rnn_8/TensorArrayV2_1/element_shape?
simple_rnn_8/TensorArrayV2_1TensorListReserve3simple_rnn_8/TensorArrayV2_1/element_shape:output:0%simple_rnn_8/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
simple_rnn_8/TensorArrayV2_1h
simple_rnn_8/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
simple_rnn_8/time?
%simple_rnn_8/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2'
%simple_rnn_8/while/maximum_iterations?
simple_rnn_8/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2!
simple_rnn_8/while/loop_counter?
simple_rnn_8/whileWhile(simple_rnn_8/while/loop_counter:output:0.simple_rnn_8/while/maximum_iterations:output:0simple_rnn_8/time:output:0%simple_rnn_8/TensorArrayV2_1:handle:0simple_rnn_8/zeros:output:0%simple_rnn_8/strided_slice_1:output:0Dsimple_rnn_8/TensorArrayUnstack/TensorListFromTensor:output_handle:0>simple_rnn_8_simple_rnn_cell_16_matmul_readvariableop_resource?simple_rnn_8_simple_rnn_cell_16_biasadd_readvariableop_resource@simple_rnn_8_simple_rnn_cell_16_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :??????????: : : : : *%
_read_only_resource_inputs
	**
body"R 
simple_rnn_8_while_body_195308**
cond"R 
simple_rnn_8_while_cond_195307*9
output_shapes(
&: : : : :??????????: : : : : *
parallel_iterations 2
simple_rnn_8/while?
=simple_rnn_8/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2?
=simple_rnn_8/TensorArrayV2Stack/TensorListStack/element_shape?
/simple_rnn_8/TensorArrayV2Stack/TensorListStackTensorListStacksimple_rnn_8/while:output:3Fsimple_rnn_8/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:x??????????*
element_dtype021
/simple_rnn_8/TensorArrayV2Stack/TensorListStack?
"simple_rnn_8/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2$
"simple_rnn_8/strided_slice_3/stack?
$simple_rnn_8/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2&
$simple_rnn_8/strided_slice_3/stack_1?
$simple_rnn_8/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$simple_rnn_8/strided_slice_3/stack_2?
simple_rnn_8/strided_slice_3StridedSlice8simple_rnn_8/TensorArrayV2Stack/TensorListStack:tensor:0+simple_rnn_8/strided_slice_3/stack:output:0-simple_rnn_8/strided_slice_3/stack_1:output:0-simple_rnn_8/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
simple_rnn_8/strided_slice_3?
simple_rnn_8/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
simple_rnn_8/transpose_1/perm?
simple_rnn_8/transpose_1	Transpose8simple_rnn_8/TensorArrayV2Stack/TensorListStack:tensor:0&simple_rnn_8/transpose_1/perm:output:0*
T0*,
_output_shapes
:?????????x?2
simple_rnn_8/transpose_1?
activation_20/ReluRelusimple_rnn_8/transpose_1:y:0*
T0*,
_output_shapes
:?????????x?2
activation_20/Reluw
dropout_8/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout_8/dropout/Const?
dropout_8/dropout/MulMul activation_20/Relu:activations:0 dropout_8/dropout/Const:output:0*
T0*,
_output_shapes
:?????????x?2
dropout_8/dropout/Mulu
dropout_8/dropout/rateConst*
_output_shapes
: *
dtype0*
valueB
 *???=2
dropout_8/dropout/rater
dropout_8/dropout/seedConst*
_output_shapes
: *
dtype0*
value	B : 2
dropout_8/dropout/seed?
dropout_8/dropout
MLCDropout activation_20/Relu:activations:0dropout_8/dropout/rate:output:0dropout_8/dropout/seed:output:0*
T0*,
_output_shapes
:?????????x?2
dropout_8/dropoutr
simple_rnn_9/ShapeShapedropout_8/dropout:output:0*
T0*
_output_shapes
:2
simple_rnn_9/Shape?
 simple_rnn_9/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 simple_rnn_9/strided_slice/stack?
"simple_rnn_9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"simple_rnn_9/strided_slice/stack_1?
"simple_rnn_9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"simple_rnn_9/strided_slice/stack_2?
simple_rnn_9/strided_sliceStridedSlicesimple_rnn_9/Shape:output:0)simple_rnn_9/strided_slice/stack:output:0+simple_rnn_9/strided_slice/stack_1:output:0+simple_rnn_9/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
simple_rnn_9/strided_slicev
simple_rnn_9/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
simple_rnn_9/zeros/mul/y?
simple_rnn_9/zeros/mulMul#simple_rnn_9/strided_slice:output:0!simple_rnn_9/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
simple_rnn_9/zeros/muly
simple_rnn_9/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
simple_rnn_9/zeros/Less/y?
simple_rnn_9/zeros/LessLesssimple_rnn_9/zeros/mul:z:0"simple_rnn_9/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
simple_rnn_9/zeros/Less|
simple_rnn_9/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
simple_rnn_9/zeros/packed/1?
simple_rnn_9/zeros/packedPack#simple_rnn_9/strided_slice:output:0$simple_rnn_9/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
simple_rnn_9/zeros/packedy
simple_rnn_9/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
simple_rnn_9/zeros/Const?
simple_rnn_9/zerosFill"simple_rnn_9/zeros/packed:output:0!simple_rnn_9/zeros/Const:output:0*
T0*'
_output_shapes
:?????????22
simple_rnn_9/zeros?
simple_rnn_9/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
simple_rnn_9/transpose/perm?
simple_rnn_9/transpose	Transposedropout_8/dropout:output:0$simple_rnn_9/transpose/perm:output:0*
T0*,
_output_shapes
:x??????????2
simple_rnn_9/transposev
simple_rnn_9/Shape_1Shapesimple_rnn_9/transpose:y:0*
T0*
_output_shapes
:2
simple_rnn_9/Shape_1?
"simple_rnn_9/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"simple_rnn_9/strided_slice_1/stack?
$simple_rnn_9/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2&
$simple_rnn_9/strided_slice_1/stack_1?
$simple_rnn_9/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$simple_rnn_9/strided_slice_1/stack_2?
simple_rnn_9/strided_slice_1StridedSlicesimple_rnn_9/Shape_1:output:0+simple_rnn_9/strided_slice_1/stack:output:0-simple_rnn_9/strided_slice_1/stack_1:output:0-simple_rnn_9/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
simple_rnn_9/strided_slice_1?
(simple_rnn_9/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2*
(simple_rnn_9/TensorArrayV2/element_shape?
simple_rnn_9/TensorArrayV2TensorListReserve1simple_rnn_9/TensorArrayV2/element_shape:output:0%simple_rnn_9/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
simple_rnn_9/TensorArrayV2?
Bsimple_rnn_9/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2D
Bsimple_rnn_9/TensorArrayUnstack/TensorListFromTensor/element_shape?
4simple_rnn_9/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorsimple_rnn_9/transpose:y:0Ksimple_rnn_9/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type026
4simple_rnn_9/TensorArrayUnstack/TensorListFromTensor?
"simple_rnn_9/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"simple_rnn_9/strided_slice_2/stack?
$simple_rnn_9/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2&
$simple_rnn_9/strided_slice_2/stack_1?
$simple_rnn_9/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$simple_rnn_9/strided_slice_2/stack_2?
simple_rnn_9/strided_slice_2StridedSlicesimple_rnn_9/transpose:y:0+simple_rnn_9/strided_slice_2/stack:output:0-simple_rnn_9/strided_slice_2/stack_1:output:0-simple_rnn_9/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
simple_rnn_9/strided_slice_2?
5simple_rnn_9/simple_rnn_cell_17/MatMul/ReadVariableOpReadVariableOp>simple_rnn_9_simple_rnn_cell_17_matmul_readvariableop_resource*
_output_shapes
:	?2*
dtype027
5simple_rnn_9/simple_rnn_cell_17/MatMul/ReadVariableOp?
&simple_rnn_9/simple_rnn_cell_17/MatMul	MLCMatMul%simple_rnn_9/strided_slice_2:output:0=simple_rnn_9/simple_rnn_cell_17/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22(
&simple_rnn_9/simple_rnn_cell_17/MatMul?
6simple_rnn_9/simple_rnn_cell_17/BiasAdd/ReadVariableOpReadVariableOp?simple_rnn_9_simple_rnn_cell_17_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype028
6simple_rnn_9/simple_rnn_cell_17/BiasAdd/ReadVariableOp?
'simple_rnn_9/simple_rnn_cell_17/BiasAddBiasAdd0simple_rnn_9/simple_rnn_cell_17/MatMul:product:0>simple_rnn_9/simple_rnn_cell_17/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22)
'simple_rnn_9/simple_rnn_cell_17/BiasAdd?
7simple_rnn_9/simple_rnn_cell_17/MatMul_1/ReadVariableOpReadVariableOp@simple_rnn_9_simple_rnn_cell_17_matmul_1_readvariableop_resource*
_output_shapes

:22*
dtype029
7simple_rnn_9/simple_rnn_cell_17/MatMul_1/ReadVariableOp?
(simple_rnn_9/simple_rnn_cell_17/MatMul_1	MLCMatMulsimple_rnn_9/zeros:output:0?simple_rnn_9/simple_rnn_cell_17/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22*
(simple_rnn_9/simple_rnn_cell_17/MatMul_1?
#simple_rnn_9/simple_rnn_cell_17/addAddV20simple_rnn_9/simple_rnn_cell_17/BiasAdd:output:02simple_rnn_9/simple_rnn_cell_17/MatMul_1:product:0*
T0*'
_output_shapes
:?????????22%
#simple_rnn_9/simple_rnn_cell_17/add?
$simple_rnn_9/simple_rnn_cell_17/TanhTanh'simple_rnn_9/simple_rnn_cell_17/add:z:0*
T0*'
_output_shapes
:?????????22&
$simple_rnn_9/simple_rnn_cell_17/Tanh?
*simple_rnn_9/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   2,
*simple_rnn_9/TensorArrayV2_1/element_shape?
simple_rnn_9/TensorArrayV2_1TensorListReserve3simple_rnn_9/TensorArrayV2_1/element_shape:output:0%simple_rnn_9/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
simple_rnn_9/TensorArrayV2_1h
simple_rnn_9/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
simple_rnn_9/time?
%simple_rnn_9/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2'
%simple_rnn_9/while/maximum_iterations?
simple_rnn_9/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2!
simple_rnn_9/while/loop_counter?
simple_rnn_9/whileWhile(simple_rnn_9/while/loop_counter:output:0.simple_rnn_9/while/maximum_iterations:output:0simple_rnn_9/time:output:0%simple_rnn_9/TensorArrayV2_1:handle:0simple_rnn_9/zeros:output:0%simple_rnn_9/strided_slice_1:output:0Dsimple_rnn_9/TensorArrayUnstack/TensorListFromTensor:output_handle:0>simple_rnn_9_simple_rnn_cell_17_matmul_readvariableop_resource?simple_rnn_9_simple_rnn_cell_17_biasadd_readvariableop_resource@simple_rnn_9_simple_rnn_cell_17_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????2: : : : : *%
_read_only_resource_inputs
	**
body"R 
simple_rnn_9_while_body_195422**
cond"R 
simple_rnn_9_while_cond_195421*8
output_shapes'
%: : : : :?????????2: : : : : *
parallel_iterations 2
simple_rnn_9/while?
=simple_rnn_9/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   2?
=simple_rnn_9/TensorArrayV2Stack/TensorListStack/element_shape?
/simple_rnn_9/TensorArrayV2Stack/TensorListStackTensorListStacksimple_rnn_9/while:output:3Fsimple_rnn_9/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:x?????????2*
element_dtype021
/simple_rnn_9/TensorArrayV2Stack/TensorListStack?
"simple_rnn_9/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2$
"simple_rnn_9/strided_slice_3/stack?
$simple_rnn_9/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2&
$simple_rnn_9/strided_slice_3/stack_1?
$simple_rnn_9/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$simple_rnn_9/strided_slice_3/stack_2?
simple_rnn_9/strided_slice_3StridedSlice8simple_rnn_9/TensorArrayV2Stack/TensorListStack:tensor:0+simple_rnn_9/strided_slice_3/stack:output:0-simple_rnn_9/strided_slice_3/stack_1:output:0-simple_rnn_9/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????2*
shrink_axis_mask2
simple_rnn_9/strided_slice_3?
simple_rnn_9/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
simple_rnn_9/transpose_1/perm?
simple_rnn_9/transpose_1	Transpose8simple_rnn_9/TensorArrayV2Stack/TensorListStack:tensor:0&simple_rnn_9/transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????x22
simple_rnn_9/transpose_1?
activation_21/ReluRelu%simple_rnn_9/strided_slice_3:output:0*
T0*'
_output_shapes
:?????????22
activation_21/Reluw
dropout_9/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout_9/dropout/Const?
dropout_9/dropout/MulMul activation_21/Relu:activations:0 dropout_9/dropout/Const:output:0*
T0*'
_output_shapes
:?????????22
dropout_9/dropout/Mulu
dropout_9/dropout/rateConst*
_output_shapes
: *
dtype0*
valueB
 *???=2
dropout_9/dropout/rater
dropout_9/dropout/seedConst*
_output_shapes
: *
dtype0*
value	B : 2
dropout_9/dropout/seed?
dropout_9/dropout
MLCDropout activation_21/Relu:activations:0dropout_9/dropout/rate:output:0dropout_9/dropout/seed:output:0*
T0*'
_output_shapes
:?????????22
dropout_9/dropout?
!dense_28/MLCMatMul/ReadVariableOpReadVariableOp*dense_28_mlcmatmul_readvariableop_resource*
_output_shapes

:2<*
dtype02#
!dense_28/MLCMatMul/ReadVariableOp?
dense_28/MLCMatMul	MLCMatMuldropout_9/dropout:output:0)dense_28/MLCMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
dense_28/MLCMatMul?
dense_28/BiasAdd/ReadVariableOpReadVariableOp(dense_28_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype02!
dense_28/BiasAdd/ReadVariableOp?
dense_28/BiasAddBiasAdddense_28/MLCMatMul:product:0'dense_28/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
dense_28/BiasAdds
dense_28/TanhTanhdense_28/BiasAdd:output:0*
T0*'
_output_shapes
:?????????<2
dense_28/Tanh?
IdentityIdentitydense_28/Tanh:y:0 ^dense_28/BiasAdd/ReadVariableOp"^dense_28/MLCMatMul/ReadVariableOp7^simple_rnn_8/simple_rnn_cell_16/BiasAdd/ReadVariableOp6^simple_rnn_8/simple_rnn_cell_16/MatMul/ReadVariableOp8^simple_rnn_8/simple_rnn_cell_16/MatMul_1/ReadVariableOp^simple_rnn_8/while7^simple_rnn_9/simple_rnn_cell_17/BiasAdd/ReadVariableOp6^simple_rnn_9/simple_rnn_cell_17/MatMul/ReadVariableOp8^simple_rnn_9/simple_rnn_cell_17/MatMul_1/ReadVariableOp^simple_rnn_9/while*
T0*'
_output_shapes
:?????????<2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:?????????x::::::::2B
dense_28/BiasAdd/ReadVariableOpdense_28/BiasAdd/ReadVariableOp2F
!dense_28/MLCMatMul/ReadVariableOp!dense_28/MLCMatMul/ReadVariableOp2p
6simple_rnn_8/simple_rnn_cell_16/BiasAdd/ReadVariableOp6simple_rnn_8/simple_rnn_cell_16/BiasAdd/ReadVariableOp2n
5simple_rnn_8/simple_rnn_cell_16/MatMul/ReadVariableOp5simple_rnn_8/simple_rnn_cell_16/MatMul/ReadVariableOp2r
7simple_rnn_8/simple_rnn_cell_16/MatMul_1/ReadVariableOp7simple_rnn_8/simple_rnn_cell_16/MatMul_1/ReadVariableOp2(
simple_rnn_8/whilesimple_rnn_8/while2p
6simple_rnn_9/simple_rnn_cell_17/BiasAdd/ReadVariableOp6simple_rnn_9/simple_rnn_cell_17/BiasAdd/ReadVariableOp2n
5simple_rnn_9/simple_rnn_cell_17/MatMul/ReadVariableOp5simple_rnn_9/simple_rnn_cell_17/MatMul/ReadVariableOp2r
7simple_rnn_9/simple_rnn_cell_17/MatMul_1/ReadVariableOp7simple_rnn_9/simple_rnn_cell_17/MatMul_1/ReadVariableOp2(
simple_rnn_9/whilesimple_rnn_9/while:S O
+
_output_shapes
:?????????x
 
_user_specified_nameinputs
?4
?
while_body_194650
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0=
9while_simple_rnn_cell_16_matmul_readvariableop_resource_0>
:while_simple_rnn_cell_16_biasadd_readvariableop_resource_0?
;while_simple_rnn_cell_16_matmul_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor;
7while_simple_rnn_cell_16_matmul_readvariableop_resource<
8while_simple_rnn_cell_16_biasadd_readvariableop_resource=
9while_simple_rnn_cell_16_matmul_1_readvariableop_resource??/while/simple_rnn_cell_16/BiasAdd/ReadVariableOp?.while/simple_rnn_cell_16/MatMul/ReadVariableOp?0while/simple_rnn_cell_16/MatMul_1/ReadVariableOp?
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
.while/simple_rnn_cell_16/MatMul/ReadVariableOpReadVariableOp9while_simple_rnn_cell_16_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype020
.while/simple_rnn_cell_16/MatMul/ReadVariableOp?
while/simple_rnn_cell_16/MatMul	MLCMatMul0while/TensorArrayV2Read/TensorListGetItem:item:06while/simple_rnn_cell_16/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2!
while/simple_rnn_cell_16/MatMul?
/while/simple_rnn_cell_16/BiasAdd/ReadVariableOpReadVariableOp:while_simple_rnn_cell_16_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype021
/while/simple_rnn_cell_16/BiasAdd/ReadVariableOp?
 while/simple_rnn_cell_16/BiasAddBiasAdd)while/simple_rnn_cell_16/MatMul:product:07while/simple_rnn_cell_16/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2"
 while/simple_rnn_cell_16/BiasAdd?
0while/simple_rnn_cell_16/MatMul_1/ReadVariableOpReadVariableOp;while_simple_rnn_cell_16_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype022
0while/simple_rnn_cell_16/MatMul_1/ReadVariableOp?
!while/simple_rnn_cell_16/MatMul_1	MLCMatMulwhile_placeholder_28while/simple_rnn_cell_16/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2#
!while/simple_rnn_cell_16/MatMul_1?
while/simple_rnn_cell_16/addAddV2)while/simple_rnn_cell_16/BiasAdd:output:0+while/simple_rnn_cell_16/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
while/simple_rnn_cell_16/add?
while/simple_rnn_cell_16/TanhTanh while/simple_rnn_cell_16/add:z:0*
T0*(
_output_shapes
:??????????2
while/simple_rnn_cell_16/Tanh?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder!while/simple_rnn_cell_16/Tanh:y:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1?
while/IdentityIdentitywhile/add_1:z:00^while/simple_rnn_cell_16/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_16/MatMul/ReadVariableOp1^while/simple_rnn_cell_16/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations0^while/simple_rnn_cell_16/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_16/MatMul/ReadVariableOp1^while/simple_rnn_cell_16/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:00^while/simple_rnn_cell_16/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_16/MatMul/ReadVariableOp1^while/simple_rnn_cell_16/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:00^while/simple_rnn_cell_16/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_16/MatMul/ReadVariableOp1^while/simple_rnn_cell_16/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity!while/simple_rnn_cell_16/Tanh:y:00^while/simple_rnn_cell_16/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_16/MatMul/ReadVariableOp1^while/simple_rnn_cell_16/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2
while/Identity_4")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"v
8while_simple_rnn_cell_16_biasadd_readvariableop_resource:while_simple_rnn_cell_16_biasadd_readvariableop_resource_0"x
9while_simple_rnn_cell_16_matmul_1_readvariableop_resource;while_simple_rnn_cell_16_matmul_1_readvariableop_resource_0"t
7while_simple_rnn_cell_16_matmul_readvariableop_resource9while_simple_rnn_cell_16_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*?
_input_shapes.
,: : : : :??????????: : :::2b
/while/simple_rnn_cell_16/BiasAdd/ReadVariableOp/while/simple_rnn_cell_16/BiasAdd/ReadVariableOp2`
.while/simple_rnn_cell_16/MatMul/ReadVariableOp.while/simple_rnn_cell_16/MatMul/ReadVariableOp2d
0while/simple_rnn_cell_16/MatMul_1/ReadVariableOp0while/simple_rnn_cell_16/MatMul_1/ReadVariableOp: 
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
: :.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?4
?
while_body_196458
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0=
9while_simple_rnn_cell_17_matmul_readvariableop_resource_0>
:while_simple_rnn_cell_17_biasadd_readvariableop_resource_0?
;while_simple_rnn_cell_17_matmul_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor;
7while_simple_rnn_cell_17_matmul_readvariableop_resource<
8while_simple_rnn_cell_17_biasadd_readvariableop_resource=
9while_simple_rnn_cell_17_matmul_1_readvariableop_resource??/while/simple_rnn_cell_17/BiasAdd/ReadVariableOp?.while/simple_rnn_cell_17/MatMul/ReadVariableOp?0while/simple_rnn_cell_17/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
.while/simple_rnn_cell_17/MatMul/ReadVariableOpReadVariableOp9while_simple_rnn_cell_17_matmul_readvariableop_resource_0*
_output_shapes
:	?2*
dtype020
.while/simple_rnn_cell_17/MatMul/ReadVariableOp?
while/simple_rnn_cell_17/MatMul	MLCMatMul0while/TensorArrayV2Read/TensorListGetItem:item:06while/simple_rnn_cell_17/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22!
while/simple_rnn_cell_17/MatMul?
/while/simple_rnn_cell_17/BiasAdd/ReadVariableOpReadVariableOp:while_simple_rnn_cell_17_biasadd_readvariableop_resource_0*
_output_shapes
:2*
dtype021
/while/simple_rnn_cell_17/BiasAdd/ReadVariableOp?
 while/simple_rnn_cell_17/BiasAddBiasAdd)while/simple_rnn_cell_17/MatMul:product:07while/simple_rnn_cell_17/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22"
 while/simple_rnn_cell_17/BiasAdd?
0while/simple_rnn_cell_17/MatMul_1/ReadVariableOpReadVariableOp;while_simple_rnn_cell_17_matmul_1_readvariableop_resource_0*
_output_shapes

:22*
dtype022
0while/simple_rnn_cell_17/MatMul_1/ReadVariableOp?
!while/simple_rnn_cell_17/MatMul_1	MLCMatMulwhile_placeholder_28while/simple_rnn_cell_17/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22#
!while/simple_rnn_cell_17/MatMul_1?
while/simple_rnn_cell_17/addAddV2)while/simple_rnn_cell_17/BiasAdd:output:0+while/simple_rnn_cell_17/MatMul_1:product:0*
T0*'
_output_shapes
:?????????22
while/simple_rnn_cell_17/add?
while/simple_rnn_cell_17/TanhTanh while/simple_rnn_cell_17/add:z:0*
T0*'
_output_shapes
:?????????22
while/simple_rnn_cell_17/Tanh?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder!while/simple_rnn_cell_17/Tanh:y:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1?
while/IdentityIdentitywhile/add_1:z:00^while/simple_rnn_cell_17/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_17/MatMul/ReadVariableOp1^while/simple_rnn_cell_17/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations0^while/simple_rnn_cell_17/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_17/MatMul/ReadVariableOp1^while/simple_rnn_cell_17/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:00^while/simple_rnn_cell_17/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_17/MatMul/ReadVariableOp1^while/simple_rnn_cell_17/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:00^while/simple_rnn_cell_17/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_17/MatMul/ReadVariableOp1^while/simple_rnn_cell_17/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity!while/simple_rnn_cell_17/Tanh:y:00^while/simple_rnn_cell_17/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_17/MatMul/ReadVariableOp1^while/simple_rnn_cell_17/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????22
while/Identity_4")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"v
8while_simple_rnn_cell_17_biasadd_readvariableop_resource:while_simple_rnn_cell_17_biasadd_readvariableop_resource_0"x
9while_simple_rnn_cell_17_matmul_1_readvariableop_resource;while_simple_rnn_cell_17_matmul_1_readvariableop_resource_0"t
7while_simple_rnn_cell_17_matmul_readvariableop_resource9while_simple_rnn_cell_17_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :?????????2: : :::2b
/while/simple_rnn_cell_17/BiasAdd/ReadVariableOp/while/simple_rnn_cell_17/BiasAdd/ReadVariableOp2`
.while/simple_rnn_cell_17/MatMul/ReadVariableOp.while/simple_rnn_cell_17/MatMul/ReadVariableOp2d
0while/simple_rnn_cell_17/MatMul_1/ReadVariableOp0while/simple_rnn_cell_17/MatMul_1/ReadVariableOp: 
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
:?????????2:

_output_shapes
: :

_output_shapes
: 
?H
?
H__inference_simple_rnn_9_layer_call_and_return_conditional_losses_194897

inputs5
1simple_rnn_cell_17_matmul_readvariableop_resource6
2simple_rnn_cell_17_biasadd_readvariableop_resource7
3simple_rnn_cell_17_matmul_1_readvariableop_resource
identity??)simple_rnn_cell_17/BiasAdd/ReadVariableOp?(simple_rnn_cell_17/MatMul/ReadVariableOp?*simple_rnn_cell_17/MatMul_1/ReadVariableOp?whileD
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
value	B :22
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
value	B :22
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
:?????????22
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm{
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:x??????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
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
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_2?
(simple_rnn_cell_17/MatMul/ReadVariableOpReadVariableOp1simple_rnn_cell_17_matmul_readvariableop_resource*
_output_shapes
:	?2*
dtype02*
(simple_rnn_cell_17/MatMul/ReadVariableOp?
simple_rnn_cell_17/MatMul	MLCMatMulstrided_slice_2:output:00simple_rnn_cell_17/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
simple_rnn_cell_17/MatMul?
)simple_rnn_cell_17/BiasAdd/ReadVariableOpReadVariableOp2simple_rnn_cell_17_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02+
)simple_rnn_cell_17/BiasAdd/ReadVariableOp?
simple_rnn_cell_17/BiasAddBiasAdd#simple_rnn_cell_17/MatMul:product:01simple_rnn_cell_17/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
simple_rnn_cell_17/BiasAdd?
*simple_rnn_cell_17/MatMul_1/ReadVariableOpReadVariableOp3simple_rnn_cell_17_matmul_1_readvariableop_resource*
_output_shapes

:22*
dtype02,
*simple_rnn_cell_17/MatMul_1/ReadVariableOp?
simple_rnn_cell_17/MatMul_1	MLCMatMulzeros:output:02simple_rnn_cell_17/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
simple_rnn_cell_17/MatMul_1?
simple_rnn_cell_17/addAddV2#simple_rnn_cell_17/BiasAdd:output:0%simple_rnn_cell_17/MatMul_1:product:0*
T0*'
_output_shapes
:?????????22
simple_rnn_cell_17/add?
simple_rnn_cell_17/TanhTanhsimple_rnn_cell_17/add:z:0*
T0*'
_output_shapes
:?????????22
simple_rnn_cell_17/Tanh?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:01simple_rnn_cell_17_matmul_readvariableop_resource2simple_rnn_cell_17_biasadd_readvariableop_resource3simple_rnn_cell_17_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????2: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_194831*
condR
while_cond_194830*8
output_shapes'
%: : : : :?????????2: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:x?????????2*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????2*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????x22
transpose_1?
IdentityIdentitystrided_slice_3:output:0*^simple_rnn_cell_17/BiasAdd/ReadVariableOp)^simple_rnn_cell_17/MatMul/ReadVariableOp+^simple_rnn_cell_17/MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????x?:::2V
)simple_rnn_cell_17/BiasAdd/ReadVariableOp)simple_rnn_cell_17/BiasAdd/ReadVariableOp2T
(simple_rnn_cell_17/MatMul/ReadVariableOp(simple_rnn_cell_17/MatMul/ReadVariableOp2X
*simple_rnn_cell_17/MatMul_1/ReadVariableOp*simple_rnn_cell_17/MatMul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:?????????x?
 
_user_specified_nameinputs
?
~
)__inference_dense_28_layer_call_fn_196846

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
:?????????<*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_28_layer_call_and_return_conditional_losses_1950902
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????<2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????2::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs
?
e
I__inference_activation_20_layer_call_and_return_conditional_losses_196271

inputs
identityS
ReluReluinputs*
T0*,
_output_shapes
:?????????x?2
Reluk
IdentityIdentityRelu:activations:0*
T0*,
_output_shapes
:?????????x?2

Identity"
identityIdentity:output:0*+
_input_shapes
:?????????x?:T P
,
_output_shapes
:?????????x?
 
_user_specified_nameinputs
?4
?
while_body_194831
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0=
9while_simple_rnn_cell_17_matmul_readvariableop_resource_0>
:while_simple_rnn_cell_17_biasadd_readvariableop_resource_0?
;while_simple_rnn_cell_17_matmul_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor;
7while_simple_rnn_cell_17_matmul_readvariableop_resource<
8while_simple_rnn_cell_17_biasadd_readvariableop_resource=
9while_simple_rnn_cell_17_matmul_1_readvariableop_resource??/while/simple_rnn_cell_17/BiasAdd/ReadVariableOp?.while/simple_rnn_cell_17/MatMul/ReadVariableOp?0while/simple_rnn_cell_17/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
.while/simple_rnn_cell_17/MatMul/ReadVariableOpReadVariableOp9while_simple_rnn_cell_17_matmul_readvariableop_resource_0*
_output_shapes
:	?2*
dtype020
.while/simple_rnn_cell_17/MatMul/ReadVariableOp?
while/simple_rnn_cell_17/MatMul	MLCMatMul0while/TensorArrayV2Read/TensorListGetItem:item:06while/simple_rnn_cell_17/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22!
while/simple_rnn_cell_17/MatMul?
/while/simple_rnn_cell_17/BiasAdd/ReadVariableOpReadVariableOp:while_simple_rnn_cell_17_biasadd_readvariableop_resource_0*
_output_shapes
:2*
dtype021
/while/simple_rnn_cell_17/BiasAdd/ReadVariableOp?
 while/simple_rnn_cell_17/BiasAddBiasAdd)while/simple_rnn_cell_17/MatMul:product:07while/simple_rnn_cell_17/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22"
 while/simple_rnn_cell_17/BiasAdd?
0while/simple_rnn_cell_17/MatMul_1/ReadVariableOpReadVariableOp;while_simple_rnn_cell_17_matmul_1_readvariableop_resource_0*
_output_shapes

:22*
dtype022
0while/simple_rnn_cell_17/MatMul_1/ReadVariableOp?
!while/simple_rnn_cell_17/MatMul_1	MLCMatMulwhile_placeholder_28while/simple_rnn_cell_17/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22#
!while/simple_rnn_cell_17/MatMul_1?
while/simple_rnn_cell_17/addAddV2)while/simple_rnn_cell_17/BiasAdd:output:0+while/simple_rnn_cell_17/MatMul_1:product:0*
T0*'
_output_shapes
:?????????22
while/simple_rnn_cell_17/add?
while/simple_rnn_cell_17/TanhTanh while/simple_rnn_cell_17/add:z:0*
T0*'
_output_shapes
:?????????22
while/simple_rnn_cell_17/Tanh?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder!while/simple_rnn_cell_17/Tanh:y:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1?
while/IdentityIdentitywhile/add_1:z:00^while/simple_rnn_cell_17/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_17/MatMul/ReadVariableOp1^while/simple_rnn_cell_17/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations0^while/simple_rnn_cell_17/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_17/MatMul/ReadVariableOp1^while/simple_rnn_cell_17/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:00^while/simple_rnn_cell_17/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_17/MatMul/ReadVariableOp1^while/simple_rnn_cell_17/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:00^while/simple_rnn_cell_17/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_17/MatMul/ReadVariableOp1^while/simple_rnn_cell_17/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity!while/simple_rnn_cell_17/Tanh:y:00^while/simple_rnn_cell_17/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_17/MatMul/ReadVariableOp1^while/simple_rnn_cell_17/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????22
while/Identity_4")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"v
8while_simple_rnn_cell_17_biasadd_readvariableop_resource:while_simple_rnn_cell_17_biasadd_readvariableop_resource_0"x
9while_simple_rnn_cell_17_matmul_1_readvariableop_resource;while_simple_rnn_cell_17_matmul_1_readvariableop_resource_0"t
7while_simple_rnn_cell_17_matmul_readvariableop_resource9while_simple_rnn_cell_17_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :?????????2: : :::2b
/while/simple_rnn_cell_17/BiasAdd/ReadVariableOp/while/simple_rnn_cell_17/BiasAdd/ReadVariableOp2`
.while/simple_rnn_cell_17/MatMul/ReadVariableOp.while/simple_rnn_cell_17/MatMul/ReadVariableOp2d
0while/simple_rnn_cell_17/MatMul_1/ReadVariableOp0while/simple_rnn_cell_17/MatMul_1/ReadVariableOp: 
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
:?????????2:

_output_shapes
: :

_output_shapes
: 
?
F
*__inference_dropout_9_layer_call_fn_196821

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_9_layer_call_and_return_conditional_losses_1950612
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????2:O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs
?
?
,sequential_24_simple_rnn_9_while_cond_193389R
Nsequential_24_simple_rnn_9_while_sequential_24_simple_rnn_9_while_loop_counterX
Tsequential_24_simple_rnn_9_while_sequential_24_simple_rnn_9_while_maximum_iterations0
,sequential_24_simple_rnn_9_while_placeholder2
.sequential_24_simple_rnn_9_while_placeholder_12
.sequential_24_simple_rnn_9_while_placeholder_2T
Psequential_24_simple_rnn_9_while_less_sequential_24_simple_rnn_9_strided_slice_1j
fsequential_24_simple_rnn_9_while_sequential_24_simple_rnn_9_while_cond_193389___redundant_placeholder0j
fsequential_24_simple_rnn_9_while_sequential_24_simple_rnn_9_while_cond_193389___redundant_placeholder1j
fsequential_24_simple_rnn_9_while_sequential_24_simple_rnn_9_while_cond_193389___redundant_placeholder2j
fsequential_24_simple_rnn_9_while_sequential_24_simple_rnn_9_while_cond_193389___redundant_placeholder3-
)sequential_24_simple_rnn_9_while_identity
?
%sequential_24/simple_rnn_9/while/LessLess,sequential_24_simple_rnn_9_while_placeholderPsequential_24_simple_rnn_9_while_less_sequential_24_simple_rnn_9_strided_slice_1*
T0*
_output_shapes
: 2'
%sequential_24/simple_rnn_9/while/Less?
)sequential_24/simple_rnn_9/while/IdentityIdentity)sequential_24/simple_rnn_9/while/Less:z:0*
T0
*
_output_shapes
: 2+
)sequential_24/simple_rnn_9/while/Identity"_
)sequential_24_simple_rnn_9_while_identity2sequential_24/simple_rnn_9/while/Identity:output:0*@
_input_shapes/
-: : : : :?????????2: ::::: 
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
:?????????2:

_output_shapes
: :

_output_shapes
:
?
?
.__inference_sequential_24_layer_call_fn_195753

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????<**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_sequential_24_layer_call_and_return_conditional_losses_1951642
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????<2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:?????????x::::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????x
 
_user_specified_nameinputs
?H
?
H__inference_simple_rnn_9_layer_call_and_return_conditional_losses_195009

inputs5
1simple_rnn_cell_17_matmul_readvariableop_resource6
2simple_rnn_cell_17_biasadd_readvariableop_resource7
3simple_rnn_cell_17_matmul_1_readvariableop_resource
identity??)simple_rnn_cell_17/BiasAdd/ReadVariableOp?(simple_rnn_cell_17/MatMul/ReadVariableOp?*simple_rnn_cell_17/MatMul_1/ReadVariableOp?whileD
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
value	B :22
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
value	B :22
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
:?????????22
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm{
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:x??????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
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
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_2?
(simple_rnn_cell_17/MatMul/ReadVariableOpReadVariableOp1simple_rnn_cell_17_matmul_readvariableop_resource*
_output_shapes
:	?2*
dtype02*
(simple_rnn_cell_17/MatMul/ReadVariableOp?
simple_rnn_cell_17/MatMul	MLCMatMulstrided_slice_2:output:00simple_rnn_cell_17/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
simple_rnn_cell_17/MatMul?
)simple_rnn_cell_17/BiasAdd/ReadVariableOpReadVariableOp2simple_rnn_cell_17_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02+
)simple_rnn_cell_17/BiasAdd/ReadVariableOp?
simple_rnn_cell_17/BiasAddBiasAdd#simple_rnn_cell_17/MatMul:product:01simple_rnn_cell_17/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
simple_rnn_cell_17/BiasAdd?
*simple_rnn_cell_17/MatMul_1/ReadVariableOpReadVariableOp3simple_rnn_cell_17_matmul_1_readvariableop_resource*
_output_shapes

:22*
dtype02,
*simple_rnn_cell_17/MatMul_1/ReadVariableOp?
simple_rnn_cell_17/MatMul_1	MLCMatMulzeros:output:02simple_rnn_cell_17/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
simple_rnn_cell_17/MatMul_1?
simple_rnn_cell_17/addAddV2#simple_rnn_cell_17/BiasAdd:output:0%simple_rnn_cell_17/MatMul_1:product:0*
T0*'
_output_shapes
:?????????22
simple_rnn_cell_17/add?
simple_rnn_cell_17/TanhTanhsimple_rnn_cell_17/add:z:0*
T0*'
_output_shapes
:?????????22
simple_rnn_cell_17/Tanh?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:01simple_rnn_cell_17_matmul_readvariableop_resource2simple_rnn_cell_17_biasadd_readvariableop_resource3simple_rnn_cell_17_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????2: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_194943*
condR
while_cond_194942*8
output_shapes'
%: : : : :?????????2: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:x?????????2*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????2*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????x22
transpose_1?
IdentityIdentitystrided_slice_3:output:0*^simple_rnn_cell_17/BiasAdd/ReadVariableOp)^simple_rnn_cell_17/MatMul/ReadVariableOp+^simple_rnn_cell_17/MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????x?:::2V
)simple_rnn_cell_17/BiasAdd/ReadVariableOp)simple_rnn_cell_17/BiasAdd/ReadVariableOp2T
(simple_rnn_cell_17/MatMul/ReadVariableOp(simple_rnn_cell_17/MatMul/ReadVariableOp2X
*simple_rnn_cell_17/MatMul_1/ReadVariableOp*simple_rnn_cell_17/MatMul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:?????????x?
 
_user_specified_nameinputs
?
?
while_cond_194416
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_194416___redundant_placeholder04
0while_while_cond_194416___redundant_placeholder14
0while_while_cond_194416___redundant_placeholder24
0while_while_cond_194416___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
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
while_identitywhile/Identity:output:0*@
_input_shapes/
-: : : : :?????????2: ::::: 
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
:?????????2:

_output_shapes
: :

_output_shapes
:
?
?
$__inference_signature_wrapper_195262
simple_rnn_8_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallsimple_rnn_8_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????<**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? **
f%R#
!__inference__wrapped_model_1934652
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????<2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:?????????x::::::::22
StatefulPartitionedCallStatefulPartitionedCall:_ [
+
_output_shapes
:?????????x
,
_user_specified_namesimple_rnn_8_input
ܘ
?
"__inference__traced_restore_197213
file_prefix$
 assignvariableop_dense_28_kernel$
 assignvariableop_1_dense_28_bias 
assignvariableop_2_adam_iter"
assignvariableop_3_adam_beta_1"
assignvariableop_4_adam_beta_2!
assignvariableop_5_adam_decay)
%assignvariableop_6_adam_learning_rate=
9assignvariableop_7_simple_rnn_8_simple_rnn_cell_16_kernelG
Cassignvariableop_8_simple_rnn_8_simple_rnn_cell_16_recurrent_kernel;
7assignvariableop_9_simple_rnn_8_simple_rnn_cell_16_bias>
:assignvariableop_10_simple_rnn_9_simple_rnn_cell_17_kernelH
Dassignvariableop_11_simple_rnn_9_simple_rnn_cell_17_recurrent_kernel<
8assignvariableop_12_simple_rnn_9_simple_rnn_cell_17_bias
assignvariableop_13_total
assignvariableop_14_count
assignvariableop_15_total_1
assignvariableop_16_count_1
assignvariableop_17_total_2
assignvariableop_18_count_2.
*assignvariableop_19_adam_dense_28_kernel_m,
(assignvariableop_20_adam_dense_28_bias_mE
Aassignvariableop_21_adam_simple_rnn_8_simple_rnn_cell_16_kernel_mO
Kassignvariableop_22_adam_simple_rnn_8_simple_rnn_cell_16_recurrent_kernel_mC
?assignvariableop_23_adam_simple_rnn_8_simple_rnn_cell_16_bias_mE
Aassignvariableop_24_adam_simple_rnn_9_simple_rnn_cell_17_kernel_mO
Kassignvariableop_25_adam_simple_rnn_9_simple_rnn_cell_17_recurrent_kernel_mC
?assignvariableop_26_adam_simple_rnn_9_simple_rnn_cell_17_bias_m.
*assignvariableop_27_adam_dense_28_kernel_v,
(assignvariableop_28_adam_dense_28_bias_vE
Aassignvariableop_29_adam_simple_rnn_8_simple_rnn_cell_16_kernel_vO
Kassignvariableop_30_adam_simple_rnn_8_simple_rnn_cell_16_recurrent_kernel_vC
?assignvariableop_31_adam_simple_rnn_8_simple_rnn_cell_16_bias_vE
Aassignvariableop_32_adam_simple_rnn_9_simple_rnn_cell_17_kernel_vO
Kassignvariableop_33_adam_simple_rnn_9_simple_rnn_cell_17_recurrent_kernel_vC
?assignvariableop_34_adam_simple_rnn_9_simple_rnn_cell_17_bias_v
identity_36??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:$*
dtype0*?
value?B?$B6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:$*
dtype0*[
valueRBP$B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::*2
dtypes(
&2$	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp assignvariableop_dense_28_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_28_biasIdentity_1:output:0"/device:CPU:0*
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
AssignVariableOp_7AssignVariableOp9assignvariableop_7_simple_rnn_8_simple_rnn_cell_16_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOpCassignvariableop_8_simple_rnn_8_simple_rnn_cell_16_recurrent_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp7assignvariableop_9_simple_rnn_8_simple_rnn_cell_16_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp:assignvariableop_10_simple_rnn_9_simple_rnn_cell_17_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOpDassignvariableop_11_simple_rnn_9_simple_rnn_cell_17_recurrent_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp8assignvariableop_12_simple_rnn_9_simple_rnn_cell_17_biasIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOpassignvariableop_13_totalIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOpassignvariableop_14_countIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOpassignvariableop_15_total_1Identity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOpassignvariableop_16_count_1Identity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOpassignvariableop_17_total_2Identity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOpassignvariableop_18_count_2Identity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp*assignvariableop_19_adam_dense_28_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp(assignvariableop_20_adam_dense_28_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOpAassignvariableop_21_adam_simple_rnn_8_simple_rnn_cell_16_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOpKassignvariableop_22_adam_simple_rnn_8_simple_rnn_cell_16_recurrent_kernel_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp?assignvariableop_23_adam_simple_rnn_8_simple_rnn_cell_16_bias_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOpAassignvariableop_24_adam_simple_rnn_9_simple_rnn_cell_17_kernel_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOpKassignvariableop_25_adam_simple_rnn_9_simple_rnn_cell_17_recurrent_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp?assignvariableop_26_adam_simple_rnn_9_simple_rnn_cell_17_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp*assignvariableop_27_adam_dense_28_kernel_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp(assignvariableop_28_adam_dense_28_bias_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOpAassignvariableop_29_adam_simple_rnn_8_simple_rnn_cell_16_kernel_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOpKassignvariableop_30_adam_simple_rnn_8_simple_rnn_cell_16_recurrent_kernel_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp?assignvariableop_31_adam_simple_rnn_8_simple_rnn_cell_16_bias_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOpAassignvariableop_32_adam_simple_rnn_9_simple_rnn_cell_17_kernel_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOpKassignvariableop_33_adam_simple_rnn_9_simple_rnn_cell_17_recurrent_kernel_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp?assignvariableop_34_adam_simple_rnn_9_simple_rnn_cell_17_bias_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_349
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_35Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_35?
Identity_36IdentityIdentity_35:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_36"#
identity_36Identity_36:output:0*?
_input_shapes?
?: :::::::::::::::::::::::::::::::::::2$
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
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342(
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
??
?

!__inference__wrapped_model_193465
simple_rnn_8_inputP
Lsequential_24_simple_rnn_8_simple_rnn_cell_16_matmul_readvariableop_resourceQ
Msequential_24_simple_rnn_8_simple_rnn_cell_16_biasadd_readvariableop_resourceR
Nsequential_24_simple_rnn_8_simple_rnn_cell_16_matmul_1_readvariableop_resourceP
Lsequential_24_simple_rnn_9_simple_rnn_cell_17_matmul_readvariableop_resourceQ
Msequential_24_simple_rnn_9_simple_rnn_cell_17_biasadd_readvariableop_resourceR
Nsequential_24_simple_rnn_9_simple_rnn_cell_17_matmul_1_readvariableop_resource<
8sequential_24_dense_28_mlcmatmul_readvariableop_resource:
6sequential_24_dense_28_biasadd_readvariableop_resource
identity??-sequential_24/dense_28/BiasAdd/ReadVariableOp?/sequential_24/dense_28/MLCMatMul/ReadVariableOp?Dsequential_24/simple_rnn_8/simple_rnn_cell_16/BiasAdd/ReadVariableOp?Csequential_24/simple_rnn_8/simple_rnn_cell_16/MatMul/ReadVariableOp?Esequential_24/simple_rnn_8/simple_rnn_cell_16/MatMul_1/ReadVariableOp? sequential_24/simple_rnn_8/while?Dsequential_24/simple_rnn_9/simple_rnn_cell_17/BiasAdd/ReadVariableOp?Csequential_24/simple_rnn_9/simple_rnn_cell_17/MatMul/ReadVariableOp?Esequential_24/simple_rnn_9/simple_rnn_cell_17/MatMul_1/ReadVariableOp? sequential_24/simple_rnn_9/while?
 sequential_24/simple_rnn_8/ShapeShapesimple_rnn_8_input*
T0*
_output_shapes
:2"
 sequential_24/simple_rnn_8/Shape?
.sequential_24/simple_rnn_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 20
.sequential_24/simple_rnn_8/strided_slice/stack?
0sequential_24/simple_rnn_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0sequential_24/simple_rnn_8/strided_slice/stack_1?
0sequential_24/simple_rnn_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0sequential_24/simple_rnn_8/strided_slice/stack_2?
(sequential_24/simple_rnn_8/strided_sliceStridedSlice)sequential_24/simple_rnn_8/Shape:output:07sequential_24/simple_rnn_8/strided_slice/stack:output:09sequential_24/simple_rnn_8/strided_slice/stack_1:output:09sequential_24/simple_rnn_8/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2*
(sequential_24/simple_rnn_8/strided_slice?
&sequential_24/simple_rnn_8/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2(
&sequential_24/simple_rnn_8/zeros/mul/y?
$sequential_24/simple_rnn_8/zeros/mulMul1sequential_24/simple_rnn_8/strided_slice:output:0/sequential_24/simple_rnn_8/zeros/mul/y:output:0*
T0*
_output_shapes
: 2&
$sequential_24/simple_rnn_8/zeros/mul?
'sequential_24/simple_rnn_8/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2)
'sequential_24/simple_rnn_8/zeros/Less/y?
%sequential_24/simple_rnn_8/zeros/LessLess(sequential_24/simple_rnn_8/zeros/mul:z:00sequential_24/simple_rnn_8/zeros/Less/y:output:0*
T0*
_output_shapes
: 2'
%sequential_24/simple_rnn_8/zeros/Less?
)sequential_24/simple_rnn_8/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2+
)sequential_24/simple_rnn_8/zeros/packed/1?
'sequential_24/simple_rnn_8/zeros/packedPack1sequential_24/simple_rnn_8/strided_slice:output:02sequential_24/simple_rnn_8/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2)
'sequential_24/simple_rnn_8/zeros/packed?
&sequential_24/simple_rnn_8/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2(
&sequential_24/simple_rnn_8/zeros/Const?
 sequential_24/simple_rnn_8/zerosFill0sequential_24/simple_rnn_8/zeros/packed:output:0/sequential_24/simple_rnn_8/zeros/Const:output:0*
T0*(
_output_shapes
:??????????2"
 sequential_24/simple_rnn_8/zeros?
)sequential_24/simple_rnn_8/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2+
)sequential_24/simple_rnn_8/transpose/perm?
$sequential_24/simple_rnn_8/transpose	Transposesimple_rnn_8_input2sequential_24/simple_rnn_8/transpose/perm:output:0*
T0*+
_output_shapes
:x?????????2&
$sequential_24/simple_rnn_8/transpose?
"sequential_24/simple_rnn_8/Shape_1Shape(sequential_24/simple_rnn_8/transpose:y:0*
T0*
_output_shapes
:2$
"sequential_24/simple_rnn_8/Shape_1?
0sequential_24/simple_rnn_8/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0sequential_24/simple_rnn_8/strided_slice_1/stack?
2sequential_24/simple_rnn_8/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2sequential_24/simple_rnn_8/strided_slice_1/stack_1?
2sequential_24/simple_rnn_8/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2sequential_24/simple_rnn_8/strided_slice_1/stack_2?
*sequential_24/simple_rnn_8/strided_slice_1StridedSlice+sequential_24/simple_rnn_8/Shape_1:output:09sequential_24/simple_rnn_8/strided_slice_1/stack:output:0;sequential_24/simple_rnn_8/strided_slice_1/stack_1:output:0;sequential_24/simple_rnn_8/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*sequential_24/simple_rnn_8/strided_slice_1?
6sequential_24/simple_rnn_8/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????28
6sequential_24/simple_rnn_8/TensorArrayV2/element_shape?
(sequential_24/simple_rnn_8/TensorArrayV2TensorListReserve?sequential_24/simple_rnn_8/TensorArrayV2/element_shape:output:03sequential_24/simple_rnn_8/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02*
(sequential_24/simple_rnn_8/TensorArrayV2?
Psequential_24/simple_rnn_8/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2R
Psequential_24/simple_rnn_8/TensorArrayUnstack/TensorListFromTensor/element_shape?
Bsequential_24/simple_rnn_8/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor(sequential_24/simple_rnn_8/transpose:y:0Ysequential_24/simple_rnn_8/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02D
Bsequential_24/simple_rnn_8/TensorArrayUnstack/TensorListFromTensor?
0sequential_24/simple_rnn_8/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0sequential_24/simple_rnn_8/strided_slice_2/stack?
2sequential_24/simple_rnn_8/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2sequential_24/simple_rnn_8/strided_slice_2/stack_1?
2sequential_24/simple_rnn_8/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2sequential_24/simple_rnn_8/strided_slice_2/stack_2?
*sequential_24/simple_rnn_8/strided_slice_2StridedSlice(sequential_24/simple_rnn_8/transpose:y:09sequential_24/simple_rnn_8/strided_slice_2/stack:output:0;sequential_24/simple_rnn_8/strided_slice_2/stack_1:output:0;sequential_24/simple_rnn_8/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2,
*sequential_24/simple_rnn_8/strided_slice_2?
Csequential_24/simple_rnn_8/simple_rnn_cell_16/MatMul/ReadVariableOpReadVariableOpLsequential_24_simple_rnn_8_simple_rnn_cell_16_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02E
Csequential_24/simple_rnn_8/simple_rnn_cell_16/MatMul/ReadVariableOp?
4sequential_24/simple_rnn_8/simple_rnn_cell_16/MatMul	MLCMatMul3sequential_24/simple_rnn_8/strided_slice_2:output:0Ksequential_24/simple_rnn_8/simple_rnn_cell_16/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????26
4sequential_24/simple_rnn_8/simple_rnn_cell_16/MatMul?
Dsequential_24/simple_rnn_8/simple_rnn_cell_16/BiasAdd/ReadVariableOpReadVariableOpMsequential_24_simple_rnn_8_simple_rnn_cell_16_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02F
Dsequential_24/simple_rnn_8/simple_rnn_cell_16/BiasAdd/ReadVariableOp?
5sequential_24/simple_rnn_8/simple_rnn_cell_16/BiasAddBiasAdd>sequential_24/simple_rnn_8/simple_rnn_cell_16/MatMul:product:0Lsequential_24/simple_rnn_8/simple_rnn_cell_16/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????27
5sequential_24/simple_rnn_8/simple_rnn_cell_16/BiasAdd?
Esequential_24/simple_rnn_8/simple_rnn_cell_16/MatMul_1/ReadVariableOpReadVariableOpNsequential_24_simple_rnn_8_simple_rnn_cell_16_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02G
Esequential_24/simple_rnn_8/simple_rnn_cell_16/MatMul_1/ReadVariableOp?
6sequential_24/simple_rnn_8/simple_rnn_cell_16/MatMul_1	MLCMatMul)sequential_24/simple_rnn_8/zeros:output:0Msequential_24/simple_rnn_8/simple_rnn_cell_16/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????28
6sequential_24/simple_rnn_8/simple_rnn_cell_16/MatMul_1?
1sequential_24/simple_rnn_8/simple_rnn_cell_16/addAddV2>sequential_24/simple_rnn_8/simple_rnn_cell_16/BiasAdd:output:0@sequential_24/simple_rnn_8/simple_rnn_cell_16/MatMul_1:product:0*
T0*(
_output_shapes
:??????????23
1sequential_24/simple_rnn_8/simple_rnn_cell_16/add?
2sequential_24/simple_rnn_8/simple_rnn_cell_16/TanhTanh5sequential_24/simple_rnn_8/simple_rnn_cell_16/add:z:0*
T0*(
_output_shapes
:??????????24
2sequential_24/simple_rnn_8/simple_rnn_cell_16/Tanh?
8sequential_24/simple_rnn_8/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2:
8sequential_24/simple_rnn_8/TensorArrayV2_1/element_shape?
*sequential_24/simple_rnn_8/TensorArrayV2_1TensorListReserveAsequential_24/simple_rnn_8/TensorArrayV2_1/element_shape:output:03sequential_24/simple_rnn_8/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02,
*sequential_24/simple_rnn_8/TensorArrayV2_1?
sequential_24/simple_rnn_8/timeConst*
_output_shapes
: *
dtype0*
value	B : 2!
sequential_24/simple_rnn_8/time?
3sequential_24/simple_rnn_8/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????25
3sequential_24/simple_rnn_8/while/maximum_iterations?
-sequential_24/simple_rnn_8/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_24/simple_rnn_8/while/loop_counter?
 sequential_24/simple_rnn_8/whileWhile6sequential_24/simple_rnn_8/while/loop_counter:output:0<sequential_24/simple_rnn_8/while/maximum_iterations:output:0(sequential_24/simple_rnn_8/time:output:03sequential_24/simple_rnn_8/TensorArrayV2_1:handle:0)sequential_24/simple_rnn_8/zeros:output:03sequential_24/simple_rnn_8/strided_slice_1:output:0Rsequential_24/simple_rnn_8/TensorArrayUnstack/TensorListFromTensor:output_handle:0Lsequential_24_simple_rnn_8_simple_rnn_cell_16_matmul_readvariableop_resourceMsequential_24_simple_rnn_8_simple_rnn_cell_16_biasadd_readvariableop_resourceNsequential_24_simple_rnn_8_simple_rnn_cell_16_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :??????????: : : : : *%
_read_only_resource_inputs
	*8
body0R.
,sequential_24_simple_rnn_8_while_body_193280*8
cond0R.
,sequential_24_simple_rnn_8_while_cond_193279*9
output_shapes(
&: : : : :??????????: : : : : *
parallel_iterations 2"
 sequential_24/simple_rnn_8/while?
Ksequential_24/simple_rnn_8/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2M
Ksequential_24/simple_rnn_8/TensorArrayV2Stack/TensorListStack/element_shape?
=sequential_24/simple_rnn_8/TensorArrayV2Stack/TensorListStackTensorListStack)sequential_24/simple_rnn_8/while:output:3Tsequential_24/simple_rnn_8/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:x??????????*
element_dtype02?
=sequential_24/simple_rnn_8/TensorArrayV2Stack/TensorListStack?
0sequential_24/simple_rnn_8/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????22
0sequential_24/simple_rnn_8/strided_slice_3/stack?
2sequential_24/simple_rnn_8/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 24
2sequential_24/simple_rnn_8/strided_slice_3/stack_1?
2sequential_24/simple_rnn_8/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2sequential_24/simple_rnn_8/strided_slice_3/stack_2?
*sequential_24/simple_rnn_8/strided_slice_3StridedSliceFsequential_24/simple_rnn_8/TensorArrayV2Stack/TensorListStack:tensor:09sequential_24/simple_rnn_8/strided_slice_3/stack:output:0;sequential_24/simple_rnn_8/strided_slice_3/stack_1:output:0;sequential_24/simple_rnn_8/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2,
*sequential_24/simple_rnn_8/strided_slice_3?
+sequential_24/simple_rnn_8/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2-
+sequential_24/simple_rnn_8/transpose_1/perm?
&sequential_24/simple_rnn_8/transpose_1	TransposeFsequential_24/simple_rnn_8/TensorArrayV2Stack/TensorListStack:tensor:04sequential_24/simple_rnn_8/transpose_1/perm:output:0*
T0*,
_output_shapes
:?????????x?2(
&sequential_24/simple_rnn_8/transpose_1?
 sequential_24/activation_20/ReluRelu*sequential_24/simple_rnn_8/transpose_1:y:0*
T0*,
_output_shapes
:?????????x?2"
 sequential_24/activation_20/Relu?
 sequential_24/dropout_8/IdentityIdentity.sequential_24/activation_20/Relu:activations:0*
T0*,
_output_shapes
:?????????x?2"
 sequential_24/dropout_8/Identity?
 sequential_24/simple_rnn_9/ShapeShape)sequential_24/dropout_8/Identity:output:0*
T0*
_output_shapes
:2"
 sequential_24/simple_rnn_9/Shape?
.sequential_24/simple_rnn_9/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 20
.sequential_24/simple_rnn_9/strided_slice/stack?
0sequential_24/simple_rnn_9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0sequential_24/simple_rnn_9/strided_slice/stack_1?
0sequential_24/simple_rnn_9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0sequential_24/simple_rnn_9/strided_slice/stack_2?
(sequential_24/simple_rnn_9/strided_sliceStridedSlice)sequential_24/simple_rnn_9/Shape:output:07sequential_24/simple_rnn_9/strided_slice/stack:output:09sequential_24/simple_rnn_9/strided_slice/stack_1:output:09sequential_24/simple_rnn_9/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2*
(sequential_24/simple_rnn_9/strided_slice?
&sequential_24/simple_rnn_9/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22(
&sequential_24/simple_rnn_9/zeros/mul/y?
$sequential_24/simple_rnn_9/zeros/mulMul1sequential_24/simple_rnn_9/strided_slice:output:0/sequential_24/simple_rnn_9/zeros/mul/y:output:0*
T0*
_output_shapes
: 2&
$sequential_24/simple_rnn_9/zeros/mul?
'sequential_24/simple_rnn_9/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2)
'sequential_24/simple_rnn_9/zeros/Less/y?
%sequential_24/simple_rnn_9/zeros/LessLess(sequential_24/simple_rnn_9/zeros/mul:z:00sequential_24/simple_rnn_9/zeros/Less/y:output:0*
T0*
_output_shapes
: 2'
%sequential_24/simple_rnn_9/zeros/Less?
)sequential_24/simple_rnn_9/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22+
)sequential_24/simple_rnn_9/zeros/packed/1?
'sequential_24/simple_rnn_9/zeros/packedPack1sequential_24/simple_rnn_9/strided_slice:output:02sequential_24/simple_rnn_9/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2)
'sequential_24/simple_rnn_9/zeros/packed?
&sequential_24/simple_rnn_9/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2(
&sequential_24/simple_rnn_9/zeros/Const?
 sequential_24/simple_rnn_9/zerosFill0sequential_24/simple_rnn_9/zeros/packed:output:0/sequential_24/simple_rnn_9/zeros/Const:output:0*
T0*'
_output_shapes
:?????????22"
 sequential_24/simple_rnn_9/zeros?
)sequential_24/simple_rnn_9/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2+
)sequential_24/simple_rnn_9/transpose/perm?
$sequential_24/simple_rnn_9/transpose	Transpose)sequential_24/dropout_8/Identity:output:02sequential_24/simple_rnn_9/transpose/perm:output:0*
T0*,
_output_shapes
:x??????????2&
$sequential_24/simple_rnn_9/transpose?
"sequential_24/simple_rnn_9/Shape_1Shape(sequential_24/simple_rnn_9/transpose:y:0*
T0*
_output_shapes
:2$
"sequential_24/simple_rnn_9/Shape_1?
0sequential_24/simple_rnn_9/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0sequential_24/simple_rnn_9/strided_slice_1/stack?
2sequential_24/simple_rnn_9/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2sequential_24/simple_rnn_9/strided_slice_1/stack_1?
2sequential_24/simple_rnn_9/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2sequential_24/simple_rnn_9/strided_slice_1/stack_2?
*sequential_24/simple_rnn_9/strided_slice_1StridedSlice+sequential_24/simple_rnn_9/Shape_1:output:09sequential_24/simple_rnn_9/strided_slice_1/stack:output:0;sequential_24/simple_rnn_9/strided_slice_1/stack_1:output:0;sequential_24/simple_rnn_9/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*sequential_24/simple_rnn_9/strided_slice_1?
6sequential_24/simple_rnn_9/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????28
6sequential_24/simple_rnn_9/TensorArrayV2/element_shape?
(sequential_24/simple_rnn_9/TensorArrayV2TensorListReserve?sequential_24/simple_rnn_9/TensorArrayV2/element_shape:output:03sequential_24/simple_rnn_9/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02*
(sequential_24/simple_rnn_9/TensorArrayV2?
Psequential_24/simple_rnn_9/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2R
Psequential_24/simple_rnn_9/TensorArrayUnstack/TensorListFromTensor/element_shape?
Bsequential_24/simple_rnn_9/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor(sequential_24/simple_rnn_9/transpose:y:0Ysequential_24/simple_rnn_9/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02D
Bsequential_24/simple_rnn_9/TensorArrayUnstack/TensorListFromTensor?
0sequential_24/simple_rnn_9/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0sequential_24/simple_rnn_9/strided_slice_2/stack?
2sequential_24/simple_rnn_9/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2sequential_24/simple_rnn_9/strided_slice_2/stack_1?
2sequential_24/simple_rnn_9/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2sequential_24/simple_rnn_9/strided_slice_2/stack_2?
*sequential_24/simple_rnn_9/strided_slice_2StridedSlice(sequential_24/simple_rnn_9/transpose:y:09sequential_24/simple_rnn_9/strided_slice_2/stack:output:0;sequential_24/simple_rnn_9/strided_slice_2/stack_1:output:0;sequential_24/simple_rnn_9/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2,
*sequential_24/simple_rnn_9/strided_slice_2?
Csequential_24/simple_rnn_9/simple_rnn_cell_17/MatMul/ReadVariableOpReadVariableOpLsequential_24_simple_rnn_9_simple_rnn_cell_17_matmul_readvariableop_resource*
_output_shapes
:	?2*
dtype02E
Csequential_24/simple_rnn_9/simple_rnn_cell_17/MatMul/ReadVariableOp?
4sequential_24/simple_rnn_9/simple_rnn_cell_17/MatMul	MLCMatMul3sequential_24/simple_rnn_9/strided_slice_2:output:0Ksequential_24/simple_rnn_9/simple_rnn_cell_17/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????226
4sequential_24/simple_rnn_9/simple_rnn_cell_17/MatMul?
Dsequential_24/simple_rnn_9/simple_rnn_cell_17/BiasAdd/ReadVariableOpReadVariableOpMsequential_24_simple_rnn_9_simple_rnn_cell_17_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02F
Dsequential_24/simple_rnn_9/simple_rnn_cell_17/BiasAdd/ReadVariableOp?
5sequential_24/simple_rnn_9/simple_rnn_cell_17/BiasAddBiasAdd>sequential_24/simple_rnn_9/simple_rnn_cell_17/MatMul:product:0Lsequential_24/simple_rnn_9/simple_rnn_cell_17/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????227
5sequential_24/simple_rnn_9/simple_rnn_cell_17/BiasAdd?
Esequential_24/simple_rnn_9/simple_rnn_cell_17/MatMul_1/ReadVariableOpReadVariableOpNsequential_24_simple_rnn_9_simple_rnn_cell_17_matmul_1_readvariableop_resource*
_output_shapes

:22*
dtype02G
Esequential_24/simple_rnn_9/simple_rnn_cell_17/MatMul_1/ReadVariableOp?
6sequential_24/simple_rnn_9/simple_rnn_cell_17/MatMul_1	MLCMatMul)sequential_24/simple_rnn_9/zeros:output:0Msequential_24/simple_rnn_9/simple_rnn_cell_17/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????228
6sequential_24/simple_rnn_9/simple_rnn_cell_17/MatMul_1?
1sequential_24/simple_rnn_9/simple_rnn_cell_17/addAddV2>sequential_24/simple_rnn_9/simple_rnn_cell_17/BiasAdd:output:0@sequential_24/simple_rnn_9/simple_rnn_cell_17/MatMul_1:product:0*
T0*'
_output_shapes
:?????????223
1sequential_24/simple_rnn_9/simple_rnn_cell_17/add?
2sequential_24/simple_rnn_9/simple_rnn_cell_17/TanhTanh5sequential_24/simple_rnn_9/simple_rnn_cell_17/add:z:0*
T0*'
_output_shapes
:?????????224
2sequential_24/simple_rnn_9/simple_rnn_cell_17/Tanh?
8sequential_24/simple_rnn_9/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   2:
8sequential_24/simple_rnn_9/TensorArrayV2_1/element_shape?
*sequential_24/simple_rnn_9/TensorArrayV2_1TensorListReserveAsequential_24/simple_rnn_9/TensorArrayV2_1/element_shape:output:03sequential_24/simple_rnn_9/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02,
*sequential_24/simple_rnn_9/TensorArrayV2_1?
sequential_24/simple_rnn_9/timeConst*
_output_shapes
: *
dtype0*
value	B : 2!
sequential_24/simple_rnn_9/time?
3sequential_24/simple_rnn_9/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????25
3sequential_24/simple_rnn_9/while/maximum_iterations?
-sequential_24/simple_rnn_9/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_24/simple_rnn_9/while/loop_counter?
 sequential_24/simple_rnn_9/whileWhile6sequential_24/simple_rnn_9/while/loop_counter:output:0<sequential_24/simple_rnn_9/while/maximum_iterations:output:0(sequential_24/simple_rnn_9/time:output:03sequential_24/simple_rnn_9/TensorArrayV2_1:handle:0)sequential_24/simple_rnn_9/zeros:output:03sequential_24/simple_rnn_9/strided_slice_1:output:0Rsequential_24/simple_rnn_9/TensorArrayUnstack/TensorListFromTensor:output_handle:0Lsequential_24_simple_rnn_9_simple_rnn_cell_17_matmul_readvariableop_resourceMsequential_24_simple_rnn_9_simple_rnn_cell_17_biasadd_readvariableop_resourceNsequential_24_simple_rnn_9_simple_rnn_cell_17_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????2: : : : : *%
_read_only_resource_inputs
	*8
body0R.
,sequential_24_simple_rnn_9_while_body_193390*8
cond0R.
,sequential_24_simple_rnn_9_while_cond_193389*8
output_shapes'
%: : : : :?????????2: : : : : *
parallel_iterations 2"
 sequential_24/simple_rnn_9/while?
Ksequential_24/simple_rnn_9/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   2M
Ksequential_24/simple_rnn_9/TensorArrayV2Stack/TensorListStack/element_shape?
=sequential_24/simple_rnn_9/TensorArrayV2Stack/TensorListStackTensorListStack)sequential_24/simple_rnn_9/while:output:3Tsequential_24/simple_rnn_9/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:x?????????2*
element_dtype02?
=sequential_24/simple_rnn_9/TensorArrayV2Stack/TensorListStack?
0sequential_24/simple_rnn_9/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????22
0sequential_24/simple_rnn_9/strided_slice_3/stack?
2sequential_24/simple_rnn_9/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 24
2sequential_24/simple_rnn_9/strided_slice_3/stack_1?
2sequential_24/simple_rnn_9/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2sequential_24/simple_rnn_9/strided_slice_3/stack_2?
*sequential_24/simple_rnn_9/strided_slice_3StridedSliceFsequential_24/simple_rnn_9/TensorArrayV2Stack/TensorListStack:tensor:09sequential_24/simple_rnn_9/strided_slice_3/stack:output:0;sequential_24/simple_rnn_9/strided_slice_3/stack_1:output:0;sequential_24/simple_rnn_9/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????2*
shrink_axis_mask2,
*sequential_24/simple_rnn_9/strided_slice_3?
+sequential_24/simple_rnn_9/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2-
+sequential_24/simple_rnn_9/transpose_1/perm?
&sequential_24/simple_rnn_9/transpose_1	TransposeFsequential_24/simple_rnn_9/TensorArrayV2Stack/TensorListStack:tensor:04sequential_24/simple_rnn_9/transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????x22(
&sequential_24/simple_rnn_9/transpose_1?
 sequential_24/activation_21/ReluRelu3sequential_24/simple_rnn_9/strided_slice_3:output:0*
T0*'
_output_shapes
:?????????22"
 sequential_24/activation_21/Relu?
 sequential_24/dropout_9/IdentityIdentity.sequential_24/activation_21/Relu:activations:0*
T0*'
_output_shapes
:?????????22"
 sequential_24/dropout_9/Identity?
/sequential_24/dense_28/MLCMatMul/ReadVariableOpReadVariableOp8sequential_24_dense_28_mlcmatmul_readvariableop_resource*
_output_shapes

:2<*
dtype021
/sequential_24/dense_28/MLCMatMul/ReadVariableOp?
 sequential_24/dense_28/MLCMatMul	MLCMatMul)sequential_24/dropout_9/Identity:output:07sequential_24/dense_28/MLCMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2"
 sequential_24/dense_28/MLCMatMul?
-sequential_24/dense_28/BiasAdd/ReadVariableOpReadVariableOp6sequential_24_dense_28_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype02/
-sequential_24/dense_28/BiasAdd/ReadVariableOp?
sequential_24/dense_28/BiasAddBiasAdd*sequential_24/dense_28/MLCMatMul:product:05sequential_24/dense_28/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2 
sequential_24/dense_28/BiasAdd?
sequential_24/dense_28/TanhTanh'sequential_24/dense_28/BiasAdd:output:0*
T0*'
_output_shapes
:?????????<2
sequential_24/dense_28/Tanh?
IdentityIdentitysequential_24/dense_28/Tanh:y:0.^sequential_24/dense_28/BiasAdd/ReadVariableOp0^sequential_24/dense_28/MLCMatMul/ReadVariableOpE^sequential_24/simple_rnn_8/simple_rnn_cell_16/BiasAdd/ReadVariableOpD^sequential_24/simple_rnn_8/simple_rnn_cell_16/MatMul/ReadVariableOpF^sequential_24/simple_rnn_8/simple_rnn_cell_16/MatMul_1/ReadVariableOp!^sequential_24/simple_rnn_8/whileE^sequential_24/simple_rnn_9/simple_rnn_cell_17/BiasAdd/ReadVariableOpD^sequential_24/simple_rnn_9/simple_rnn_cell_17/MatMul/ReadVariableOpF^sequential_24/simple_rnn_9/simple_rnn_cell_17/MatMul_1/ReadVariableOp!^sequential_24/simple_rnn_9/while*
T0*'
_output_shapes
:?????????<2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:?????????x::::::::2^
-sequential_24/dense_28/BiasAdd/ReadVariableOp-sequential_24/dense_28/BiasAdd/ReadVariableOp2b
/sequential_24/dense_28/MLCMatMul/ReadVariableOp/sequential_24/dense_28/MLCMatMul/ReadVariableOp2?
Dsequential_24/simple_rnn_8/simple_rnn_cell_16/BiasAdd/ReadVariableOpDsequential_24/simple_rnn_8/simple_rnn_cell_16/BiasAdd/ReadVariableOp2?
Csequential_24/simple_rnn_8/simple_rnn_cell_16/MatMul/ReadVariableOpCsequential_24/simple_rnn_8/simple_rnn_cell_16/MatMul/ReadVariableOp2?
Esequential_24/simple_rnn_8/simple_rnn_cell_16/MatMul_1/ReadVariableOpEsequential_24/simple_rnn_8/simple_rnn_cell_16/MatMul_1/ReadVariableOp2D
 sequential_24/simple_rnn_8/while sequential_24/simple_rnn_8/while2?
Dsequential_24/simple_rnn_9/simple_rnn_cell_17/BiasAdd/ReadVariableOpDsequential_24/simple_rnn_9/simple_rnn_cell_17/BiasAdd/ReadVariableOp2?
Csequential_24/simple_rnn_9/simple_rnn_cell_17/MatMul/ReadVariableOpCsequential_24/simple_rnn_9/simple_rnn_cell_17/MatMul/ReadVariableOp2?
Esequential_24/simple_rnn_9/simple_rnn_cell_17/MatMul_1/ReadVariableOpEsequential_24/simple_rnn_9/simple_rnn_cell_17/MatMul_1/ReadVariableOp2D
 sequential_24/simple_rnn_9/while sequential_24/simple_rnn_9/while:_ [
+
_output_shapes
:?????????x
,
_user_specified_namesimple_rnn_8_input
?C
?
simple_rnn_8_while_body_1953086
2simple_rnn_8_while_simple_rnn_8_while_loop_counter<
8simple_rnn_8_while_simple_rnn_8_while_maximum_iterations"
simple_rnn_8_while_placeholder$
 simple_rnn_8_while_placeholder_1$
 simple_rnn_8_while_placeholder_25
1simple_rnn_8_while_simple_rnn_8_strided_slice_1_0q
msimple_rnn_8_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_8_tensorarrayunstack_tensorlistfromtensor_0J
Fsimple_rnn_8_while_simple_rnn_cell_16_matmul_readvariableop_resource_0K
Gsimple_rnn_8_while_simple_rnn_cell_16_biasadd_readvariableop_resource_0L
Hsimple_rnn_8_while_simple_rnn_cell_16_matmul_1_readvariableop_resource_0
simple_rnn_8_while_identity!
simple_rnn_8_while_identity_1!
simple_rnn_8_while_identity_2!
simple_rnn_8_while_identity_3!
simple_rnn_8_while_identity_43
/simple_rnn_8_while_simple_rnn_8_strided_slice_1o
ksimple_rnn_8_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_8_tensorarrayunstack_tensorlistfromtensorH
Dsimple_rnn_8_while_simple_rnn_cell_16_matmul_readvariableop_resourceI
Esimple_rnn_8_while_simple_rnn_cell_16_biasadd_readvariableop_resourceJ
Fsimple_rnn_8_while_simple_rnn_cell_16_matmul_1_readvariableop_resource??<simple_rnn_8/while/simple_rnn_cell_16/BiasAdd/ReadVariableOp?;simple_rnn_8/while/simple_rnn_cell_16/MatMul/ReadVariableOp?=simple_rnn_8/while/simple_rnn_cell_16/MatMul_1/ReadVariableOp?
Dsimple_rnn_8/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2F
Dsimple_rnn_8/while/TensorArrayV2Read/TensorListGetItem/element_shape?
6simple_rnn_8/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemmsimple_rnn_8_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_8_tensorarrayunstack_tensorlistfromtensor_0simple_rnn_8_while_placeholderMsimple_rnn_8/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype028
6simple_rnn_8/while/TensorArrayV2Read/TensorListGetItem?
;simple_rnn_8/while/simple_rnn_cell_16/MatMul/ReadVariableOpReadVariableOpFsimple_rnn_8_while_simple_rnn_cell_16_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype02=
;simple_rnn_8/while/simple_rnn_cell_16/MatMul/ReadVariableOp?
,simple_rnn_8/while/simple_rnn_cell_16/MatMul	MLCMatMul=simple_rnn_8/while/TensorArrayV2Read/TensorListGetItem:item:0Csimple_rnn_8/while/simple_rnn_cell_16/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2.
,simple_rnn_8/while/simple_rnn_cell_16/MatMul?
<simple_rnn_8/while/simple_rnn_cell_16/BiasAdd/ReadVariableOpReadVariableOpGsimple_rnn_8_while_simple_rnn_cell_16_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02>
<simple_rnn_8/while/simple_rnn_cell_16/BiasAdd/ReadVariableOp?
-simple_rnn_8/while/simple_rnn_cell_16/BiasAddBiasAdd6simple_rnn_8/while/simple_rnn_cell_16/MatMul:product:0Dsimple_rnn_8/while/simple_rnn_cell_16/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2/
-simple_rnn_8/while/simple_rnn_cell_16/BiasAdd?
=simple_rnn_8/while/simple_rnn_cell_16/MatMul_1/ReadVariableOpReadVariableOpHsimple_rnn_8_while_simple_rnn_cell_16_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02?
=simple_rnn_8/while/simple_rnn_cell_16/MatMul_1/ReadVariableOp?
.simple_rnn_8/while/simple_rnn_cell_16/MatMul_1	MLCMatMul simple_rnn_8_while_placeholder_2Esimple_rnn_8/while/simple_rnn_cell_16/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????20
.simple_rnn_8/while/simple_rnn_cell_16/MatMul_1?
)simple_rnn_8/while/simple_rnn_cell_16/addAddV26simple_rnn_8/while/simple_rnn_cell_16/BiasAdd:output:08simple_rnn_8/while/simple_rnn_cell_16/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2+
)simple_rnn_8/while/simple_rnn_cell_16/add?
*simple_rnn_8/while/simple_rnn_cell_16/TanhTanh-simple_rnn_8/while/simple_rnn_cell_16/add:z:0*
T0*(
_output_shapes
:??????????2,
*simple_rnn_8/while/simple_rnn_cell_16/Tanh?
7simple_rnn_8/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem simple_rnn_8_while_placeholder_1simple_rnn_8_while_placeholder.simple_rnn_8/while/simple_rnn_cell_16/Tanh:y:0*
_output_shapes
: *
element_dtype029
7simple_rnn_8/while/TensorArrayV2Write/TensorListSetItemv
simple_rnn_8/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
simple_rnn_8/while/add/y?
simple_rnn_8/while/addAddV2simple_rnn_8_while_placeholder!simple_rnn_8/while/add/y:output:0*
T0*
_output_shapes
: 2
simple_rnn_8/while/addz
simple_rnn_8/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
simple_rnn_8/while/add_1/y?
simple_rnn_8/while/add_1AddV22simple_rnn_8_while_simple_rnn_8_while_loop_counter#simple_rnn_8/while/add_1/y:output:0*
T0*
_output_shapes
: 2
simple_rnn_8/while/add_1?
simple_rnn_8/while/IdentityIdentitysimple_rnn_8/while/add_1:z:0=^simple_rnn_8/while/simple_rnn_cell_16/BiasAdd/ReadVariableOp<^simple_rnn_8/while/simple_rnn_cell_16/MatMul/ReadVariableOp>^simple_rnn_8/while/simple_rnn_cell_16/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
simple_rnn_8/while/Identity?
simple_rnn_8/while/Identity_1Identity8simple_rnn_8_while_simple_rnn_8_while_maximum_iterations=^simple_rnn_8/while/simple_rnn_cell_16/BiasAdd/ReadVariableOp<^simple_rnn_8/while/simple_rnn_cell_16/MatMul/ReadVariableOp>^simple_rnn_8/while/simple_rnn_cell_16/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
simple_rnn_8/while/Identity_1?
simple_rnn_8/while/Identity_2Identitysimple_rnn_8/while/add:z:0=^simple_rnn_8/while/simple_rnn_cell_16/BiasAdd/ReadVariableOp<^simple_rnn_8/while/simple_rnn_cell_16/MatMul/ReadVariableOp>^simple_rnn_8/while/simple_rnn_cell_16/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
simple_rnn_8/while/Identity_2?
simple_rnn_8/while/Identity_3IdentityGsimple_rnn_8/while/TensorArrayV2Write/TensorListSetItem:output_handle:0=^simple_rnn_8/while/simple_rnn_cell_16/BiasAdd/ReadVariableOp<^simple_rnn_8/while/simple_rnn_cell_16/MatMul/ReadVariableOp>^simple_rnn_8/while/simple_rnn_cell_16/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
simple_rnn_8/while/Identity_3?
simple_rnn_8/while/Identity_4Identity.simple_rnn_8/while/simple_rnn_cell_16/Tanh:y:0=^simple_rnn_8/while/simple_rnn_cell_16/BiasAdd/ReadVariableOp<^simple_rnn_8/while/simple_rnn_cell_16/MatMul/ReadVariableOp>^simple_rnn_8/while/simple_rnn_cell_16/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2
simple_rnn_8/while/Identity_4"C
simple_rnn_8_while_identity$simple_rnn_8/while/Identity:output:0"G
simple_rnn_8_while_identity_1&simple_rnn_8/while/Identity_1:output:0"G
simple_rnn_8_while_identity_2&simple_rnn_8/while/Identity_2:output:0"G
simple_rnn_8_while_identity_3&simple_rnn_8/while/Identity_3:output:0"G
simple_rnn_8_while_identity_4&simple_rnn_8/while/Identity_4:output:0"d
/simple_rnn_8_while_simple_rnn_8_strided_slice_11simple_rnn_8_while_simple_rnn_8_strided_slice_1_0"?
Esimple_rnn_8_while_simple_rnn_cell_16_biasadd_readvariableop_resourceGsimple_rnn_8_while_simple_rnn_cell_16_biasadd_readvariableop_resource_0"?
Fsimple_rnn_8_while_simple_rnn_cell_16_matmul_1_readvariableop_resourceHsimple_rnn_8_while_simple_rnn_cell_16_matmul_1_readvariableop_resource_0"?
Dsimple_rnn_8_while_simple_rnn_cell_16_matmul_readvariableop_resourceFsimple_rnn_8_while_simple_rnn_cell_16_matmul_readvariableop_resource_0"?
ksimple_rnn_8_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_8_tensorarrayunstack_tensorlistfromtensormsimple_rnn_8_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_8_tensorarrayunstack_tensorlistfromtensor_0*?
_input_shapes.
,: : : : :??????????: : :::2|
<simple_rnn_8/while/simple_rnn_cell_16/BiasAdd/ReadVariableOp<simple_rnn_8/while/simple_rnn_cell_16/BiasAdd/ReadVariableOp2z
;simple_rnn_8/while/simple_rnn_cell_16/MatMul/ReadVariableOp;simple_rnn_8/while/simple_rnn_cell_16/MatMul/ReadVariableOp2~
=simple_rnn_8/while/simple_rnn_cell_16/MatMul_1/ReadVariableOp=simple_rnn_8/while/simple_rnn_cell_16/MatMul_1/ReadVariableOp: 
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
: :.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?
?
while_cond_195819
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_195819___redundant_placeholder04
0while_while_cond_195819___redundant_placeholder14
0while_while_cond_195819___redundant_placeholder24
0while_while_cond_195819___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
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
while_identitywhile/Identity:output:0*A
_input_shapes0
.: : : : :??????????: ::::: 
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
: :.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?
?
while_cond_194830
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_194830___redundant_placeholder04
0while_while_cond_194830___redundant_placeholder14
0while_while_cond_194830___redundant_placeholder24
0while_while_cond_194830___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
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
while_identitywhile/Identity:output:0*@
_input_shapes/
-: : : : :?????????2: ::::: 
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
:?????????2:

_output_shapes
: :

_output_shapes
:
?
?
-__inference_simple_rnn_9_layer_call_fn_196546

inputs
unknown
	unknown_0
	unknown_1
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_simple_rnn_9_layer_call_and_return_conditional_losses_1950092
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????x?:::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:?????????x?
 
_user_specified_nameinputs
?
?
while_cond_193787
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_193787___redundant_placeholder04
0while_while_cond_193787___redundant_placeholder14
0while_while_cond_193787___redundant_placeholder24
0while_while_cond_193787___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
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
while_identitywhile/Identity:output:0*A
_input_shapes0
.: : : : :??????????: ::::: 
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
: :.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?H
?
H__inference_simple_rnn_9_layer_call_and_return_conditional_losses_196658
inputs_05
1simple_rnn_cell_17_matmul_readvariableop_resource6
2simple_rnn_cell_17_biasadd_readvariableop_resource7
3simple_rnn_cell_17_matmul_1_readvariableop_resource
identity??)simple_rnn_cell_17/BiasAdd/ReadVariableOp?(simple_rnn_cell_17/MatMul/ReadVariableOp?*simple_rnn_cell_17/MatMul_1/ReadVariableOp?whileF
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
value	B :22
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
value	B :22
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
:?????????22
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*5
_output_shapes#
!:???????????????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
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
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_2?
(simple_rnn_cell_17/MatMul/ReadVariableOpReadVariableOp1simple_rnn_cell_17_matmul_readvariableop_resource*
_output_shapes
:	?2*
dtype02*
(simple_rnn_cell_17/MatMul/ReadVariableOp?
simple_rnn_cell_17/MatMul	MLCMatMulstrided_slice_2:output:00simple_rnn_cell_17/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
simple_rnn_cell_17/MatMul?
)simple_rnn_cell_17/BiasAdd/ReadVariableOpReadVariableOp2simple_rnn_cell_17_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02+
)simple_rnn_cell_17/BiasAdd/ReadVariableOp?
simple_rnn_cell_17/BiasAddBiasAdd#simple_rnn_cell_17/MatMul:product:01simple_rnn_cell_17/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
simple_rnn_cell_17/BiasAdd?
*simple_rnn_cell_17/MatMul_1/ReadVariableOpReadVariableOp3simple_rnn_cell_17_matmul_1_readvariableop_resource*
_output_shapes

:22*
dtype02,
*simple_rnn_cell_17/MatMul_1/ReadVariableOp?
simple_rnn_cell_17/MatMul_1	MLCMatMulzeros:output:02simple_rnn_cell_17/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
simple_rnn_cell_17/MatMul_1?
simple_rnn_cell_17/addAddV2#simple_rnn_cell_17/BiasAdd:output:0%simple_rnn_cell_17/MatMul_1:product:0*
T0*'
_output_shapes
:?????????22
simple_rnn_cell_17/add?
simple_rnn_cell_17/TanhTanhsimple_rnn_cell_17/add:z:0*
T0*'
_output_shapes
:?????????22
simple_rnn_cell_17/Tanh?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:01simple_rnn_cell_17_matmul_readvariableop_resource2simple_rnn_cell_17_biasadd_readvariableop_resource3simple_rnn_cell_17_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????2: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_196592*
condR
while_cond_196591*8
output_shapes'
%: : : : :?????????2: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????2*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????2*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????22
transpose_1?
IdentityIdentitystrided_slice_3:output:0*^simple_rnn_cell_17/BiasAdd/ReadVariableOp)^simple_rnn_cell_17/MatMul/ReadVariableOp+^simple_rnn_cell_17/MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*@
_input_shapes/
-:???????????????????:::2V
)simple_rnn_cell_17/BiasAdd/ReadVariableOp)simple_rnn_cell_17/BiasAdd/ReadVariableOp2T
(simple_rnn_cell_17/MatMul/ReadVariableOp(simple_rnn_cell_17/MatMul/ReadVariableOp2X
*simple_rnn_cell_17/MatMul_1/ReadVariableOp*simple_rnn_cell_17/MatMul_1/ReadVariableOp2
whilewhile:_ [
5
_output_shapes#
!:???????????????????
"
_user_specified_name
inputs/0
?4
?
while_body_196346
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0=
9while_simple_rnn_cell_17_matmul_readvariableop_resource_0>
:while_simple_rnn_cell_17_biasadd_readvariableop_resource_0?
;while_simple_rnn_cell_17_matmul_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor;
7while_simple_rnn_cell_17_matmul_readvariableop_resource<
8while_simple_rnn_cell_17_biasadd_readvariableop_resource=
9while_simple_rnn_cell_17_matmul_1_readvariableop_resource??/while/simple_rnn_cell_17/BiasAdd/ReadVariableOp?.while/simple_rnn_cell_17/MatMul/ReadVariableOp?0while/simple_rnn_cell_17/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
.while/simple_rnn_cell_17/MatMul/ReadVariableOpReadVariableOp9while_simple_rnn_cell_17_matmul_readvariableop_resource_0*
_output_shapes
:	?2*
dtype020
.while/simple_rnn_cell_17/MatMul/ReadVariableOp?
while/simple_rnn_cell_17/MatMul	MLCMatMul0while/TensorArrayV2Read/TensorListGetItem:item:06while/simple_rnn_cell_17/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22!
while/simple_rnn_cell_17/MatMul?
/while/simple_rnn_cell_17/BiasAdd/ReadVariableOpReadVariableOp:while_simple_rnn_cell_17_biasadd_readvariableop_resource_0*
_output_shapes
:2*
dtype021
/while/simple_rnn_cell_17/BiasAdd/ReadVariableOp?
 while/simple_rnn_cell_17/BiasAddBiasAdd)while/simple_rnn_cell_17/MatMul:product:07while/simple_rnn_cell_17/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22"
 while/simple_rnn_cell_17/BiasAdd?
0while/simple_rnn_cell_17/MatMul_1/ReadVariableOpReadVariableOp;while_simple_rnn_cell_17_matmul_1_readvariableop_resource_0*
_output_shapes

:22*
dtype022
0while/simple_rnn_cell_17/MatMul_1/ReadVariableOp?
!while/simple_rnn_cell_17/MatMul_1	MLCMatMulwhile_placeholder_28while/simple_rnn_cell_17/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22#
!while/simple_rnn_cell_17/MatMul_1?
while/simple_rnn_cell_17/addAddV2)while/simple_rnn_cell_17/BiasAdd:output:0+while/simple_rnn_cell_17/MatMul_1:product:0*
T0*'
_output_shapes
:?????????22
while/simple_rnn_cell_17/add?
while/simple_rnn_cell_17/TanhTanh while/simple_rnn_cell_17/add:z:0*
T0*'
_output_shapes
:?????????22
while/simple_rnn_cell_17/Tanh?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder!while/simple_rnn_cell_17/Tanh:y:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1?
while/IdentityIdentitywhile/add_1:z:00^while/simple_rnn_cell_17/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_17/MatMul/ReadVariableOp1^while/simple_rnn_cell_17/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations0^while/simple_rnn_cell_17/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_17/MatMul/ReadVariableOp1^while/simple_rnn_cell_17/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:00^while/simple_rnn_cell_17/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_17/MatMul/ReadVariableOp1^while/simple_rnn_cell_17/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:00^while/simple_rnn_cell_17/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_17/MatMul/ReadVariableOp1^while/simple_rnn_cell_17/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity!while/simple_rnn_cell_17/Tanh:y:00^while/simple_rnn_cell_17/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_17/MatMul/ReadVariableOp1^while/simple_rnn_cell_17/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????22
while/Identity_4")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"v
8while_simple_rnn_cell_17_biasadd_readvariableop_resource:while_simple_rnn_cell_17_biasadd_readvariableop_resource_0"x
9while_simple_rnn_cell_17_matmul_1_readvariableop_resource;while_simple_rnn_cell_17_matmul_1_readvariableop_resource_0"t
7while_simple_rnn_cell_17_matmul_readvariableop_resource9while_simple_rnn_cell_17_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :?????????2: : :::2b
/while/simple_rnn_cell_17/BiasAdd/ReadVariableOp/while/simple_rnn_cell_17/BiasAdd/ReadVariableOp2`
.while/simple_rnn_cell_17/MatMul/ReadVariableOp.while/simple_rnn_cell_17/MatMul/ReadVariableOp2d
0while/simple_rnn_cell_17/MatMul_1/ReadVariableOp0while/simple_rnn_cell_17/MatMul_1/ReadVariableOp: 
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
:?????????2:

_output_shapes
: :

_output_shapes
: 
?
?
-__inference_simple_rnn_8_layer_call_fn_196020
inputs_0
unknown
	unknown_0
	unknown_1
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:???????????????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_simple_rnn_8_layer_call_and_return_conditional_losses_1939682
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*5
_output_shapes#
!:???????????????????2

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
?4
?
while_body_194943
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0=
9while_simple_rnn_cell_17_matmul_readvariableop_resource_0>
:while_simple_rnn_cell_17_biasadd_readvariableop_resource_0?
;while_simple_rnn_cell_17_matmul_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor;
7while_simple_rnn_cell_17_matmul_readvariableop_resource<
8while_simple_rnn_cell_17_biasadd_readvariableop_resource=
9while_simple_rnn_cell_17_matmul_1_readvariableop_resource??/while/simple_rnn_cell_17/BiasAdd/ReadVariableOp?.while/simple_rnn_cell_17/MatMul/ReadVariableOp?0while/simple_rnn_cell_17/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
.while/simple_rnn_cell_17/MatMul/ReadVariableOpReadVariableOp9while_simple_rnn_cell_17_matmul_readvariableop_resource_0*
_output_shapes
:	?2*
dtype020
.while/simple_rnn_cell_17/MatMul/ReadVariableOp?
while/simple_rnn_cell_17/MatMul	MLCMatMul0while/TensorArrayV2Read/TensorListGetItem:item:06while/simple_rnn_cell_17/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22!
while/simple_rnn_cell_17/MatMul?
/while/simple_rnn_cell_17/BiasAdd/ReadVariableOpReadVariableOp:while_simple_rnn_cell_17_biasadd_readvariableop_resource_0*
_output_shapes
:2*
dtype021
/while/simple_rnn_cell_17/BiasAdd/ReadVariableOp?
 while/simple_rnn_cell_17/BiasAddBiasAdd)while/simple_rnn_cell_17/MatMul:product:07while/simple_rnn_cell_17/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22"
 while/simple_rnn_cell_17/BiasAdd?
0while/simple_rnn_cell_17/MatMul_1/ReadVariableOpReadVariableOp;while_simple_rnn_cell_17_matmul_1_readvariableop_resource_0*
_output_shapes

:22*
dtype022
0while/simple_rnn_cell_17/MatMul_1/ReadVariableOp?
!while/simple_rnn_cell_17/MatMul_1	MLCMatMulwhile_placeholder_28while/simple_rnn_cell_17/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22#
!while/simple_rnn_cell_17/MatMul_1?
while/simple_rnn_cell_17/addAddV2)while/simple_rnn_cell_17/BiasAdd:output:0+while/simple_rnn_cell_17/MatMul_1:product:0*
T0*'
_output_shapes
:?????????22
while/simple_rnn_cell_17/add?
while/simple_rnn_cell_17/TanhTanh while/simple_rnn_cell_17/add:z:0*
T0*'
_output_shapes
:?????????22
while/simple_rnn_cell_17/Tanh?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder!while/simple_rnn_cell_17/Tanh:y:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1?
while/IdentityIdentitywhile/add_1:z:00^while/simple_rnn_cell_17/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_17/MatMul/ReadVariableOp1^while/simple_rnn_cell_17/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations0^while/simple_rnn_cell_17/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_17/MatMul/ReadVariableOp1^while/simple_rnn_cell_17/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:00^while/simple_rnn_cell_17/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_17/MatMul/ReadVariableOp1^while/simple_rnn_cell_17/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:00^while/simple_rnn_cell_17/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_17/MatMul/ReadVariableOp1^while/simple_rnn_cell_17/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity!while/simple_rnn_cell_17/Tanh:y:00^while/simple_rnn_cell_17/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_17/MatMul/ReadVariableOp1^while/simple_rnn_cell_17/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????22
while/Identity_4")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"v
8while_simple_rnn_cell_17_biasadd_readvariableop_resource:while_simple_rnn_cell_17_biasadd_readvariableop_resource_0"x
9while_simple_rnn_cell_17_matmul_1_readvariableop_resource;while_simple_rnn_cell_17_matmul_1_readvariableop_resource_0"t
7while_simple_rnn_cell_17_matmul_readvariableop_resource9while_simple_rnn_cell_17_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :?????????2: : :::2b
/while/simple_rnn_cell_17/BiasAdd/ReadVariableOp/while/simple_rnn_cell_17/BiasAdd/ReadVariableOp2`
.while/simple_rnn_cell_17/MatMul/ReadVariableOp.while/simple_rnn_cell_17/MatMul/ReadVariableOp2d
0while/simple_rnn_cell_17/MatMul_1/ReadVariableOp0while/simple_rnn_cell_17/MatMul_1/ReadVariableOp: 
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
:?????????2:

_output_shapes
: :

_output_shapes
: 
?	
?
3__inference_simple_rnn_cell_16_layer_call_fn_196908

inputs
states_0
unknown
	unknown_0
	unknown_1
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:??????????:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_simple_rnn_cell_16_layer_call_and_return_conditional_losses_1935312
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*F
_input_shapes5
3:?????????:??????????:::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/0
?
?
-__inference_simple_rnn_9_layer_call_fn_196781
inputs_0
unknown
	unknown_0
	unknown_1
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_simple_rnn_9_layer_call_and_return_conditional_losses_1943632
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*@
_input_shapes/
-:???????????????????:::22
StatefulPartitionedCallStatefulPartitionedCall:_ [
5
_output_shapes#
!:???????????????????
"
_user_specified_name
inputs/0
?
?
while_cond_195931
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_195931___redundant_placeholder04
0while_while_cond_195931___redundant_placeholder14
0while_while_cond_195931___redundant_placeholder24
0while_while_cond_195931___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
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
while_identitywhile/Identity:output:0*A
_input_shapes0
.: : : : :??????????: ::::: 
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
: :.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?
?
-__inference_simple_rnn_8_layer_call_fn_196266

inputs
unknown
	unknown_0
	unknown_1
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????x?*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_simple_rnn_8_layer_call_and_return_conditional_losses_1947162
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:?????????x?2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????x:::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????x
 
_user_specified_nameinputs
?
?
while_cond_194942
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_194942___redundant_placeholder04
0while_while_cond_194942___redundant_placeholder14
0while_while_cond_194942___redundant_placeholder24
0while_while_cond_194942___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
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
while_identitywhile/Identity:output:0*@
_input_shapes/
-: : : : :?????????2: ::::: 
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
:?????????2:

_output_shapes
: :

_output_shapes
:
?

?
simple_rnn_8_while_cond_1955466
2simple_rnn_8_while_simple_rnn_8_while_loop_counter<
8simple_rnn_8_while_simple_rnn_8_while_maximum_iterations"
simple_rnn_8_while_placeholder$
 simple_rnn_8_while_placeholder_1$
 simple_rnn_8_while_placeholder_28
4simple_rnn_8_while_less_simple_rnn_8_strided_slice_1N
Jsimple_rnn_8_while_simple_rnn_8_while_cond_195546___redundant_placeholder0N
Jsimple_rnn_8_while_simple_rnn_8_while_cond_195546___redundant_placeholder1N
Jsimple_rnn_8_while_simple_rnn_8_while_cond_195546___redundant_placeholder2N
Jsimple_rnn_8_while_simple_rnn_8_while_cond_195546___redundant_placeholder3
simple_rnn_8_while_identity
?
simple_rnn_8/while/LessLesssimple_rnn_8_while_placeholder4simple_rnn_8_while_less_simple_rnn_8_strided_slice_1*
T0*
_output_shapes
: 2
simple_rnn_8/while/Less?
simple_rnn_8/while/IdentityIdentitysimple_rnn_8/while/Less:z:0*
T0
*
_output_shapes
: 2
simple_rnn_8/while/Identity"C
simple_rnn_8_while_identity$simple_rnn_8/while/Identity:output:0*A
_input_shapes0
.: : : : :??????????: ::::: 
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
: :.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?4
?
while_body_195820
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0=
9while_simple_rnn_cell_16_matmul_readvariableop_resource_0>
:while_simple_rnn_cell_16_biasadd_readvariableop_resource_0?
;while_simple_rnn_cell_16_matmul_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor;
7while_simple_rnn_cell_16_matmul_readvariableop_resource<
8while_simple_rnn_cell_16_biasadd_readvariableop_resource=
9while_simple_rnn_cell_16_matmul_1_readvariableop_resource??/while/simple_rnn_cell_16/BiasAdd/ReadVariableOp?.while/simple_rnn_cell_16/MatMul/ReadVariableOp?0while/simple_rnn_cell_16/MatMul_1/ReadVariableOp?
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
.while/simple_rnn_cell_16/MatMul/ReadVariableOpReadVariableOp9while_simple_rnn_cell_16_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype020
.while/simple_rnn_cell_16/MatMul/ReadVariableOp?
while/simple_rnn_cell_16/MatMul	MLCMatMul0while/TensorArrayV2Read/TensorListGetItem:item:06while/simple_rnn_cell_16/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2!
while/simple_rnn_cell_16/MatMul?
/while/simple_rnn_cell_16/BiasAdd/ReadVariableOpReadVariableOp:while_simple_rnn_cell_16_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype021
/while/simple_rnn_cell_16/BiasAdd/ReadVariableOp?
 while/simple_rnn_cell_16/BiasAddBiasAdd)while/simple_rnn_cell_16/MatMul:product:07while/simple_rnn_cell_16/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2"
 while/simple_rnn_cell_16/BiasAdd?
0while/simple_rnn_cell_16/MatMul_1/ReadVariableOpReadVariableOp;while_simple_rnn_cell_16_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype022
0while/simple_rnn_cell_16/MatMul_1/ReadVariableOp?
!while/simple_rnn_cell_16/MatMul_1	MLCMatMulwhile_placeholder_28while/simple_rnn_cell_16/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2#
!while/simple_rnn_cell_16/MatMul_1?
while/simple_rnn_cell_16/addAddV2)while/simple_rnn_cell_16/BiasAdd:output:0+while/simple_rnn_cell_16/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
while/simple_rnn_cell_16/add?
while/simple_rnn_cell_16/TanhTanh while/simple_rnn_cell_16/add:z:0*
T0*(
_output_shapes
:??????????2
while/simple_rnn_cell_16/Tanh?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder!while/simple_rnn_cell_16/Tanh:y:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1?
while/IdentityIdentitywhile/add_1:z:00^while/simple_rnn_cell_16/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_16/MatMul/ReadVariableOp1^while/simple_rnn_cell_16/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations0^while/simple_rnn_cell_16/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_16/MatMul/ReadVariableOp1^while/simple_rnn_cell_16/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:00^while/simple_rnn_cell_16/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_16/MatMul/ReadVariableOp1^while/simple_rnn_cell_16/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:00^while/simple_rnn_cell_16/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_16/MatMul/ReadVariableOp1^while/simple_rnn_cell_16/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity!while/simple_rnn_cell_16/Tanh:y:00^while/simple_rnn_cell_16/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_16/MatMul/ReadVariableOp1^while/simple_rnn_cell_16/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2
while/Identity_4")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"v
8while_simple_rnn_cell_16_biasadd_readvariableop_resource:while_simple_rnn_cell_16_biasadd_readvariableop_resource_0"x
9while_simple_rnn_cell_16_matmul_1_readvariableop_resource;while_simple_rnn_cell_16_matmul_1_readvariableop_resource_0"t
7while_simple_rnn_cell_16_matmul_readvariableop_resource9while_simple_rnn_cell_16_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*?
_input_shapes.
,: : : : :??????????: : :::2b
/while/simple_rnn_cell_16/BiasAdd/ReadVariableOp/while/simple_rnn_cell_16/BiasAdd/ReadVariableOp2`
.while/simple_rnn_cell_16/MatMul/ReadVariableOp.while/simple_rnn_cell_16/MatMul/ReadVariableOp2d
0while/simple_rnn_cell_16/MatMul_1/ReadVariableOp0while/simple_rnn_cell_16/MatMul_1/ReadVariableOp: 
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
: :.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
??
?
I__inference_sequential_24_layer_call_and_return_conditional_losses_195732

inputsB
>simple_rnn_8_simple_rnn_cell_16_matmul_readvariableop_resourceC
?simple_rnn_8_simple_rnn_cell_16_biasadd_readvariableop_resourceD
@simple_rnn_8_simple_rnn_cell_16_matmul_1_readvariableop_resourceB
>simple_rnn_9_simple_rnn_cell_17_matmul_readvariableop_resourceC
?simple_rnn_9_simple_rnn_cell_17_biasadd_readvariableop_resourceD
@simple_rnn_9_simple_rnn_cell_17_matmul_1_readvariableop_resource.
*dense_28_mlcmatmul_readvariableop_resource,
(dense_28_biasadd_readvariableop_resource
identity??dense_28/BiasAdd/ReadVariableOp?!dense_28/MLCMatMul/ReadVariableOp?6simple_rnn_8/simple_rnn_cell_16/BiasAdd/ReadVariableOp?5simple_rnn_8/simple_rnn_cell_16/MatMul/ReadVariableOp?7simple_rnn_8/simple_rnn_cell_16/MatMul_1/ReadVariableOp?simple_rnn_8/while?6simple_rnn_9/simple_rnn_cell_17/BiasAdd/ReadVariableOp?5simple_rnn_9/simple_rnn_cell_17/MatMul/ReadVariableOp?7simple_rnn_9/simple_rnn_cell_17/MatMul_1/ReadVariableOp?simple_rnn_9/while^
simple_rnn_8/ShapeShapeinputs*
T0*
_output_shapes
:2
simple_rnn_8/Shape?
 simple_rnn_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 simple_rnn_8/strided_slice/stack?
"simple_rnn_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"simple_rnn_8/strided_slice/stack_1?
"simple_rnn_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"simple_rnn_8/strided_slice/stack_2?
simple_rnn_8/strided_sliceStridedSlicesimple_rnn_8/Shape:output:0)simple_rnn_8/strided_slice/stack:output:0+simple_rnn_8/strided_slice/stack_1:output:0+simple_rnn_8/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
simple_rnn_8/strided_slicew
simple_rnn_8/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
simple_rnn_8/zeros/mul/y?
simple_rnn_8/zeros/mulMul#simple_rnn_8/strided_slice:output:0!simple_rnn_8/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
simple_rnn_8/zeros/muly
simple_rnn_8/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
simple_rnn_8/zeros/Less/y?
simple_rnn_8/zeros/LessLesssimple_rnn_8/zeros/mul:z:0"simple_rnn_8/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
simple_rnn_8/zeros/Less}
simple_rnn_8/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
simple_rnn_8/zeros/packed/1?
simple_rnn_8/zeros/packedPack#simple_rnn_8/strided_slice:output:0$simple_rnn_8/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
simple_rnn_8/zeros/packedy
simple_rnn_8/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
simple_rnn_8/zeros/Const?
simple_rnn_8/zerosFill"simple_rnn_8/zeros/packed:output:0!simple_rnn_8/zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
simple_rnn_8/zeros?
simple_rnn_8/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
simple_rnn_8/transpose/perm?
simple_rnn_8/transpose	Transposeinputs$simple_rnn_8/transpose/perm:output:0*
T0*+
_output_shapes
:x?????????2
simple_rnn_8/transposev
simple_rnn_8/Shape_1Shapesimple_rnn_8/transpose:y:0*
T0*
_output_shapes
:2
simple_rnn_8/Shape_1?
"simple_rnn_8/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"simple_rnn_8/strided_slice_1/stack?
$simple_rnn_8/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2&
$simple_rnn_8/strided_slice_1/stack_1?
$simple_rnn_8/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$simple_rnn_8/strided_slice_1/stack_2?
simple_rnn_8/strided_slice_1StridedSlicesimple_rnn_8/Shape_1:output:0+simple_rnn_8/strided_slice_1/stack:output:0-simple_rnn_8/strided_slice_1/stack_1:output:0-simple_rnn_8/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
simple_rnn_8/strided_slice_1?
(simple_rnn_8/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2*
(simple_rnn_8/TensorArrayV2/element_shape?
simple_rnn_8/TensorArrayV2TensorListReserve1simple_rnn_8/TensorArrayV2/element_shape:output:0%simple_rnn_8/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
simple_rnn_8/TensorArrayV2?
Bsimple_rnn_8/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2D
Bsimple_rnn_8/TensorArrayUnstack/TensorListFromTensor/element_shape?
4simple_rnn_8/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorsimple_rnn_8/transpose:y:0Ksimple_rnn_8/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type026
4simple_rnn_8/TensorArrayUnstack/TensorListFromTensor?
"simple_rnn_8/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"simple_rnn_8/strided_slice_2/stack?
$simple_rnn_8/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2&
$simple_rnn_8/strided_slice_2/stack_1?
$simple_rnn_8/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$simple_rnn_8/strided_slice_2/stack_2?
simple_rnn_8/strided_slice_2StridedSlicesimple_rnn_8/transpose:y:0+simple_rnn_8/strided_slice_2/stack:output:0-simple_rnn_8/strided_slice_2/stack_1:output:0-simple_rnn_8/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
simple_rnn_8/strided_slice_2?
5simple_rnn_8/simple_rnn_cell_16/MatMul/ReadVariableOpReadVariableOp>simple_rnn_8_simple_rnn_cell_16_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype027
5simple_rnn_8/simple_rnn_cell_16/MatMul/ReadVariableOp?
&simple_rnn_8/simple_rnn_cell_16/MatMul	MLCMatMul%simple_rnn_8/strided_slice_2:output:0=simple_rnn_8/simple_rnn_cell_16/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2(
&simple_rnn_8/simple_rnn_cell_16/MatMul?
6simple_rnn_8/simple_rnn_cell_16/BiasAdd/ReadVariableOpReadVariableOp?simple_rnn_8_simple_rnn_cell_16_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype028
6simple_rnn_8/simple_rnn_cell_16/BiasAdd/ReadVariableOp?
'simple_rnn_8/simple_rnn_cell_16/BiasAddBiasAdd0simple_rnn_8/simple_rnn_cell_16/MatMul:product:0>simple_rnn_8/simple_rnn_cell_16/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2)
'simple_rnn_8/simple_rnn_cell_16/BiasAdd?
7simple_rnn_8/simple_rnn_cell_16/MatMul_1/ReadVariableOpReadVariableOp@simple_rnn_8_simple_rnn_cell_16_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype029
7simple_rnn_8/simple_rnn_cell_16/MatMul_1/ReadVariableOp?
(simple_rnn_8/simple_rnn_cell_16/MatMul_1	MLCMatMulsimple_rnn_8/zeros:output:0?simple_rnn_8/simple_rnn_cell_16/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2*
(simple_rnn_8/simple_rnn_cell_16/MatMul_1?
#simple_rnn_8/simple_rnn_cell_16/addAddV20simple_rnn_8/simple_rnn_cell_16/BiasAdd:output:02simple_rnn_8/simple_rnn_cell_16/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2%
#simple_rnn_8/simple_rnn_cell_16/add?
$simple_rnn_8/simple_rnn_cell_16/TanhTanh'simple_rnn_8/simple_rnn_cell_16/add:z:0*
T0*(
_output_shapes
:??????????2&
$simple_rnn_8/simple_rnn_cell_16/Tanh?
*simple_rnn_8/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2,
*simple_rnn_8/TensorArrayV2_1/element_shape?
simple_rnn_8/TensorArrayV2_1TensorListReserve3simple_rnn_8/TensorArrayV2_1/element_shape:output:0%simple_rnn_8/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
simple_rnn_8/TensorArrayV2_1h
simple_rnn_8/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
simple_rnn_8/time?
%simple_rnn_8/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2'
%simple_rnn_8/while/maximum_iterations?
simple_rnn_8/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2!
simple_rnn_8/while/loop_counter?
simple_rnn_8/whileWhile(simple_rnn_8/while/loop_counter:output:0.simple_rnn_8/while/maximum_iterations:output:0simple_rnn_8/time:output:0%simple_rnn_8/TensorArrayV2_1:handle:0simple_rnn_8/zeros:output:0%simple_rnn_8/strided_slice_1:output:0Dsimple_rnn_8/TensorArrayUnstack/TensorListFromTensor:output_handle:0>simple_rnn_8_simple_rnn_cell_16_matmul_readvariableop_resource?simple_rnn_8_simple_rnn_cell_16_biasadd_readvariableop_resource@simple_rnn_8_simple_rnn_cell_16_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :??????????: : : : : *%
_read_only_resource_inputs
	**
body"R 
simple_rnn_8_while_body_195547**
cond"R 
simple_rnn_8_while_cond_195546*9
output_shapes(
&: : : : :??????????: : : : : *
parallel_iterations 2
simple_rnn_8/while?
=simple_rnn_8/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2?
=simple_rnn_8/TensorArrayV2Stack/TensorListStack/element_shape?
/simple_rnn_8/TensorArrayV2Stack/TensorListStackTensorListStacksimple_rnn_8/while:output:3Fsimple_rnn_8/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:x??????????*
element_dtype021
/simple_rnn_8/TensorArrayV2Stack/TensorListStack?
"simple_rnn_8/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2$
"simple_rnn_8/strided_slice_3/stack?
$simple_rnn_8/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2&
$simple_rnn_8/strided_slice_3/stack_1?
$simple_rnn_8/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$simple_rnn_8/strided_slice_3/stack_2?
simple_rnn_8/strided_slice_3StridedSlice8simple_rnn_8/TensorArrayV2Stack/TensorListStack:tensor:0+simple_rnn_8/strided_slice_3/stack:output:0-simple_rnn_8/strided_slice_3/stack_1:output:0-simple_rnn_8/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
simple_rnn_8/strided_slice_3?
simple_rnn_8/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
simple_rnn_8/transpose_1/perm?
simple_rnn_8/transpose_1	Transpose8simple_rnn_8/TensorArrayV2Stack/TensorListStack:tensor:0&simple_rnn_8/transpose_1/perm:output:0*
T0*,
_output_shapes
:?????????x?2
simple_rnn_8/transpose_1?
activation_20/ReluRelusimple_rnn_8/transpose_1:y:0*
T0*,
_output_shapes
:?????????x?2
activation_20/Relu?
dropout_8/IdentityIdentity activation_20/Relu:activations:0*
T0*,
_output_shapes
:?????????x?2
dropout_8/Identitys
simple_rnn_9/ShapeShapedropout_8/Identity:output:0*
T0*
_output_shapes
:2
simple_rnn_9/Shape?
 simple_rnn_9/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 simple_rnn_9/strided_slice/stack?
"simple_rnn_9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"simple_rnn_9/strided_slice/stack_1?
"simple_rnn_9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"simple_rnn_9/strided_slice/stack_2?
simple_rnn_9/strided_sliceStridedSlicesimple_rnn_9/Shape:output:0)simple_rnn_9/strided_slice/stack:output:0+simple_rnn_9/strided_slice/stack_1:output:0+simple_rnn_9/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
simple_rnn_9/strided_slicev
simple_rnn_9/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
simple_rnn_9/zeros/mul/y?
simple_rnn_9/zeros/mulMul#simple_rnn_9/strided_slice:output:0!simple_rnn_9/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
simple_rnn_9/zeros/muly
simple_rnn_9/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
simple_rnn_9/zeros/Less/y?
simple_rnn_9/zeros/LessLesssimple_rnn_9/zeros/mul:z:0"simple_rnn_9/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
simple_rnn_9/zeros/Less|
simple_rnn_9/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
simple_rnn_9/zeros/packed/1?
simple_rnn_9/zeros/packedPack#simple_rnn_9/strided_slice:output:0$simple_rnn_9/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
simple_rnn_9/zeros/packedy
simple_rnn_9/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
simple_rnn_9/zeros/Const?
simple_rnn_9/zerosFill"simple_rnn_9/zeros/packed:output:0!simple_rnn_9/zeros/Const:output:0*
T0*'
_output_shapes
:?????????22
simple_rnn_9/zeros?
simple_rnn_9/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
simple_rnn_9/transpose/perm?
simple_rnn_9/transpose	Transposedropout_8/Identity:output:0$simple_rnn_9/transpose/perm:output:0*
T0*,
_output_shapes
:x??????????2
simple_rnn_9/transposev
simple_rnn_9/Shape_1Shapesimple_rnn_9/transpose:y:0*
T0*
_output_shapes
:2
simple_rnn_9/Shape_1?
"simple_rnn_9/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"simple_rnn_9/strided_slice_1/stack?
$simple_rnn_9/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2&
$simple_rnn_9/strided_slice_1/stack_1?
$simple_rnn_9/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$simple_rnn_9/strided_slice_1/stack_2?
simple_rnn_9/strided_slice_1StridedSlicesimple_rnn_9/Shape_1:output:0+simple_rnn_9/strided_slice_1/stack:output:0-simple_rnn_9/strided_slice_1/stack_1:output:0-simple_rnn_9/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
simple_rnn_9/strided_slice_1?
(simple_rnn_9/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2*
(simple_rnn_9/TensorArrayV2/element_shape?
simple_rnn_9/TensorArrayV2TensorListReserve1simple_rnn_9/TensorArrayV2/element_shape:output:0%simple_rnn_9/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
simple_rnn_9/TensorArrayV2?
Bsimple_rnn_9/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2D
Bsimple_rnn_9/TensorArrayUnstack/TensorListFromTensor/element_shape?
4simple_rnn_9/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorsimple_rnn_9/transpose:y:0Ksimple_rnn_9/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type026
4simple_rnn_9/TensorArrayUnstack/TensorListFromTensor?
"simple_rnn_9/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"simple_rnn_9/strided_slice_2/stack?
$simple_rnn_9/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2&
$simple_rnn_9/strided_slice_2/stack_1?
$simple_rnn_9/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$simple_rnn_9/strided_slice_2/stack_2?
simple_rnn_9/strided_slice_2StridedSlicesimple_rnn_9/transpose:y:0+simple_rnn_9/strided_slice_2/stack:output:0-simple_rnn_9/strided_slice_2/stack_1:output:0-simple_rnn_9/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
simple_rnn_9/strided_slice_2?
5simple_rnn_9/simple_rnn_cell_17/MatMul/ReadVariableOpReadVariableOp>simple_rnn_9_simple_rnn_cell_17_matmul_readvariableop_resource*
_output_shapes
:	?2*
dtype027
5simple_rnn_9/simple_rnn_cell_17/MatMul/ReadVariableOp?
&simple_rnn_9/simple_rnn_cell_17/MatMul	MLCMatMul%simple_rnn_9/strided_slice_2:output:0=simple_rnn_9/simple_rnn_cell_17/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22(
&simple_rnn_9/simple_rnn_cell_17/MatMul?
6simple_rnn_9/simple_rnn_cell_17/BiasAdd/ReadVariableOpReadVariableOp?simple_rnn_9_simple_rnn_cell_17_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype028
6simple_rnn_9/simple_rnn_cell_17/BiasAdd/ReadVariableOp?
'simple_rnn_9/simple_rnn_cell_17/BiasAddBiasAdd0simple_rnn_9/simple_rnn_cell_17/MatMul:product:0>simple_rnn_9/simple_rnn_cell_17/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22)
'simple_rnn_9/simple_rnn_cell_17/BiasAdd?
7simple_rnn_9/simple_rnn_cell_17/MatMul_1/ReadVariableOpReadVariableOp@simple_rnn_9_simple_rnn_cell_17_matmul_1_readvariableop_resource*
_output_shapes

:22*
dtype029
7simple_rnn_9/simple_rnn_cell_17/MatMul_1/ReadVariableOp?
(simple_rnn_9/simple_rnn_cell_17/MatMul_1	MLCMatMulsimple_rnn_9/zeros:output:0?simple_rnn_9/simple_rnn_cell_17/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22*
(simple_rnn_9/simple_rnn_cell_17/MatMul_1?
#simple_rnn_9/simple_rnn_cell_17/addAddV20simple_rnn_9/simple_rnn_cell_17/BiasAdd:output:02simple_rnn_9/simple_rnn_cell_17/MatMul_1:product:0*
T0*'
_output_shapes
:?????????22%
#simple_rnn_9/simple_rnn_cell_17/add?
$simple_rnn_9/simple_rnn_cell_17/TanhTanh'simple_rnn_9/simple_rnn_cell_17/add:z:0*
T0*'
_output_shapes
:?????????22&
$simple_rnn_9/simple_rnn_cell_17/Tanh?
*simple_rnn_9/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   2,
*simple_rnn_9/TensorArrayV2_1/element_shape?
simple_rnn_9/TensorArrayV2_1TensorListReserve3simple_rnn_9/TensorArrayV2_1/element_shape:output:0%simple_rnn_9/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
simple_rnn_9/TensorArrayV2_1h
simple_rnn_9/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
simple_rnn_9/time?
%simple_rnn_9/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2'
%simple_rnn_9/while/maximum_iterations?
simple_rnn_9/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2!
simple_rnn_9/while/loop_counter?
simple_rnn_9/whileWhile(simple_rnn_9/while/loop_counter:output:0.simple_rnn_9/while/maximum_iterations:output:0simple_rnn_9/time:output:0%simple_rnn_9/TensorArrayV2_1:handle:0simple_rnn_9/zeros:output:0%simple_rnn_9/strided_slice_1:output:0Dsimple_rnn_9/TensorArrayUnstack/TensorListFromTensor:output_handle:0>simple_rnn_9_simple_rnn_cell_17_matmul_readvariableop_resource?simple_rnn_9_simple_rnn_cell_17_biasadd_readvariableop_resource@simple_rnn_9_simple_rnn_cell_17_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????2: : : : : *%
_read_only_resource_inputs
	**
body"R 
simple_rnn_9_while_body_195657**
cond"R 
simple_rnn_9_while_cond_195656*8
output_shapes'
%: : : : :?????????2: : : : : *
parallel_iterations 2
simple_rnn_9/while?
=simple_rnn_9/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   2?
=simple_rnn_9/TensorArrayV2Stack/TensorListStack/element_shape?
/simple_rnn_9/TensorArrayV2Stack/TensorListStackTensorListStacksimple_rnn_9/while:output:3Fsimple_rnn_9/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:x?????????2*
element_dtype021
/simple_rnn_9/TensorArrayV2Stack/TensorListStack?
"simple_rnn_9/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2$
"simple_rnn_9/strided_slice_3/stack?
$simple_rnn_9/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2&
$simple_rnn_9/strided_slice_3/stack_1?
$simple_rnn_9/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$simple_rnn_9/strided_slice_3/stack_2?
simple_rnn_9/strided_slice_3StridedSlice8simple_rnn_9/TensorArrayV2Stack/TensorListStack:tensor:0+simple_rnn_9/strided_slice_3/stack:output:0-simple_rnn_9/strided_slice_3/stack_1:output:0-simple_rnn_9/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????2*
shrink_axis_mask2
simple_rnn_9/strided_slice_3?
simple_rnn_9/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
simple_rnn_9/transpose_1/perm?
simple_rnn_9/transpose_1	Transpose8simple_rnn_9/TensorArrayV2Stack/TensorListStack:tensor:0&simple_rnn_9/transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????x22
simple_rnn_9/transpose_1?
activation_21/ReluRelu%simple_rnn_9/strided_slice_3:output:0*
T0*'
_output_shapes
:?????????22
activation_21/Relu?
dropout_9/IdentityIdentity activation_21/Relu:activations:0*
T0*'
_output_shapes
:?????????22
dropout_9/Identity?
!dense_28/MLCMatMul/ReadVariableOpReadVariableOp*dense_28_mlcmatmul_readvariableop_resource*
_output_shapes

:2<*
dtype02#
!dense_28/MLCMatMul/ReadVariableOp?
dense_28/MLCMatMul	MLCMatMuldropout_9/Identity:output:0)dense_28/MLCMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
dense_28/MLCMatMul?
dense_28/BiasAdd/ReadVariableOpReadVariableOp(dense_28_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype02!
dense_28/BiasAdd/ReadVariableOp?
dense_28/BiasAddBiasAdddense_28/MLCMatMul:product:0'dense_28/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
dense_28/BiasAdds
dense_28/TanhTanhdense_28/BiasAdd:output:0*
T0*'
_output_shapes
:?????????<2
dense_28/Tanh?
IdentityIdentitydense_28/Tanh:y:0 ^dense_28/BiasAdd/ReadVariableOp"^dense_28/MLCMatMul/ReadVariableOp7^simple_rnn_8/simple_rnn_cell_16/BiasAdd/ReadVariableOp6^simple_rnn_8/simple_rnn_cell_16/MatMul/ReadVariableOp8^simple_rnn_8/simple_rnn_cell_16/MatMul_1/ReadVariableOp^simple_rnn_8/while7^simple_rnn_9/simple_rnn_cell_17/BiasAdd/ReadVariableOp6^simple_rnn_9/simple_rnn_cell_17/MatMul/ReadVariableOp8^simple_rnn_9/simple_rnn_cell_17/MatMul_1/ReadVariableOp^simple_rnn_9/while*
T0*'
_output_shapes
:?????????<2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:?????????x::::::::2B
dense_28/BiasAdd/ReadVariableOpdense_28/BiasAdd/ReadVariableOp2F
!dense_28/MLCMatMul/ReadVariableOp!dense_28/MLCMatMul/ReadVariableOp2p
6simple_rnn_8/simple_rnn_cell_16/BiasAdd/ReadVariableOp6simple_rnn_8/simple_rnn_cell_16/BiasAdd/ReadVariableOp2n
5simple_rnn_8/simple_rnn_cell_16/MatMul/ReadVariableOp5simple_rnn_8/simple_rnn_cell_16/MatMul/ReadVariableOp2r
7simple_rnn_8/simple_rnn_cell_16/MatMul_1/ReadVariableOp7simple_rnn_8/simple_rnn_cell_16/MatMul_1/ReadVariableOp2(
simple_rnn_8/whilesimple_rnn_8/while2p
6simple_rnn_9/simple_rnn_cell_17/BiasAdd/ReadVariableOp6simple_rnn_9/simple_rnn_cell_17/BiasAdd/ReadVariableOp2n
5simple_rnn_9/simple_rnn_cell_17/MatMul/ReadVariableOp5simple_rnn_9/simple_rnn_cell_17/MatMul/ReadVariableOp2r
7simple_rnn_9/simple_rnn_cell_17/MatMul_1/ReadVariableOp7simple_rnn_9/simple_rnn_cell_17/MatMul_1/ReadVariableOp2(
simple_rnn_9/whilesimple_rnn_9/while:S O
+
_output_shapes
:?????????x
 
_user_specified_nameinputs
?
?
while_cond_196457
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_196457___redundant_placeholder04
0while_while_cond_196457___redundant_placeholder14
0while_while_cond_196457___redundant_placeholder24
0while_while_cond_196457___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
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
while_identitywhile/Identity:output:0*@
_input_shapes/
-: : : : :?????????2: ::::: 
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
:?????????2:

_output_shapes
: :

_output_shapes
:
?

?
D__inference_dense_28_layer_call_and_return_conditional_losses_195090

inputs%
!mlcmatmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MLCMatMul/ReadVariableOp?
MLCMatMul/ReadVariableOpReadVariableOp!mlcmatmul_readvariableop_resource*
_output_shapes

:2<*
dtype02
MLCMatMul/ReadVariableOp
	MLCMatMul	MLCMatMulinputs MLCMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
	MLCMatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:<*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMLCMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2	
BiasAddX
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:?????????<2
Tanh?
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MLCMatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????<2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????2::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
MLCMatMul/ReadVariableOpMLCMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs
?
?
N__inference_simple_rnn_cell_17_layer_call_and_return_conditional_losses_196925

inputs
states_0"
matmul_readvariableop_resource#
biasadd_readvariableop_resource$
 matmul_1_readvariableop_resource
identity

identity_1??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?MatMul_1/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?2*
dtype02
MatMul/ReadVariableOpv
MatMul	MLCMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:2*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22	
BiasAdd?
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:22*
dtype02
MatMul_1/ReadVariableOp~
MatMul_1	MLCMatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22

MatMul_1k
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*'
_output_shapes
:?????????22
addO
TanhTanhadd:z:0*
T0*'
_output_shapes
:?????????22
Tanh?
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????22

Identity?

Identity_1IdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????22

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*F
_input_shapes5
3:??????????:?????????2:::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:QM
'
_output_shapes
:?????????2
"
_user_specified_name
states/0
?
?
while_cond_193904
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_193904___redundant_placeholder04
0while_while_cond_193904___redundant_placeholder14
0while_while_cond_193904___redundant_placeholder24
0while_while_cond_193904___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
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
while_identitywhile/Identity:output:0*A
_input_shapes0
.: : : : :??????????: ::::: 
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
: :.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?#
?
while_body_194300
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0%
!while_simple_rnn_cell_17_194322_0%
!while_simple_rnn_cell_17_194324_0%
!while_simple_rnn_cell_17_194326_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor#
while_simple_rnn_cell_17_194322#
while_simple_rnn_cell_17_194324#
while_simple_rnn_cell_17_194326??0while/simple_rnn_cell_17/StatefulPartitionedCall?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
0while/simple_rnn_cell_17/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2!while_simple_rnn_cell_17_194322_0!while_simple_rnn_cell_17_194324_0!while_simple_rnn_cell_17_194326_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????2:?????????2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_simple_rnn_cell_17_layer_call_and_return_conditional_losses_19402622
0while/simple_rnn_cell_17/StatefulPartitionedCall?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder9while/simple_rnn_cell_17/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1?
while/IdentityIdentitywhile/add_1:z:01^while/simple_rnn_cell_17/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations1^while/simple_rnn_cell_17/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:01^while/simple_rnn_cell_17/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:01^while/simple_rnn_cell_17/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity9while/simple_rnn_cell_17/StatefulPartitionedCall:output:11^while/simple_rnn_cell_17/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????22
while/Identity_4")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"D
while_simple_rnn_cell_17_194322!while_simple_rnn_cell_17_194322_0"D
while_simple_rnn_cell_17_194324!while_simple_rnn_cell_17_194324_0"D
while_simple_rnn_cell_17_194326!while_simple_rnn_cell_17_194326_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :?????????2: : :::2d
0while/simple_rnn_cell_17/StatefulPartitionedCall0while/simple_rnn_cell_17/StatefulPartitionedCall: 
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
:?????????2:

_output_shapes
: :

_output_shapes
: 
?H
?
H__inference_simple_rnn_8_layer_call_and_return_conditional_losses_195998
inputs_05
1simple_rnn_cell_16_matmul_readvariableop_resource6
2simple_rnn_cell_16_biasadd_readvariableop_resource7
3simple_rnn_cell_16_matmul_1_readvariableop_resource
identity??)simple_rnn_cell_16/BiasAdd/ReadVariableOp?(simple_rnn_cell_16/MatMul/ReadVariableOp?*simple_rnn_cell_16/MatMul_1/ReadVariableOp?whileF
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
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
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

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
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
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
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
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
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
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_2?
(simple_rnn_cell_16/MatMul/ReadVariableOpReadVariableOp1simple_rnn_cell_16_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02*
(simple_rnn_cell_16/MatMul/ReadVariableOp?
simple_rnn_cell_16/MatMul	MLCMatMulstrided_slice_2:output:00simple_rnn_cell_16/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
simple_rnn_cell_16/MatMul?
)simple_rnn_cell_16/BiasAdd/ReadVariableOpReadVariableOp2simple_rnn_cell_16_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)simple_rnn_cell_16/BiasAdd/ReadVariableOp?
simple_rnn_cell_16/BiasAddBiasAdd#simple_rnn_cell_16/MatMul:product:01simple_rnn_cell_16/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
simple_rnn_cell_16/BiasAdd?
*simple_rnn_cell_16/MatMul_1/ReadVariableOpReadVariableOp3simple_rnn_cell_16_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02,
*simple_rnn_cell_16/MatMul_1/ReadVariableOp?
simple_rnn_cell_16/MatMul_1	MLCMatMulzeros:output:02simple_rnn_cell_16/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
simple_rnn_cell_16/MatMul_1?
simple_rnn_cell_16/addAddV2#simple_rnn_cell_16/BiasAdd:output:0%simple_rnn_cell_16/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
simple_rnn_cell_16/add?
simple_rnn_cell_16/TanhTanhsimple_rnn_cell_16/add:z:0*
T0*(
_output_shapes
:??????????2
simple_rnn_cell_16/Tanh?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:01simple_rnn_cell_16_matmul_readvariableop_resource2simple_rnn_cell_16_biasadd_readvariableop_resource3simple_rnn_cell_16_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :??????????: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_195932*
condR
while_cond_195931*9
output_shapes(
&: : : : :??????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:???????????????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:???????????????????2
transpose_1?
IdentityIdentitytranspose_1:y:0*^simple_rnn_cell_16/BiasAdd/ReadVariableOp)^simple_rnn_cell_16/MatMul/ReadVariableOp+^simple_rnn_cell_16/MatMul_1/ReadVariableOp^while*
T0*5
_output_shapes#
!:???????????????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????:::2V
)simple_rnn_cell_16/BiasAdd/ReadVariableOp)simple_rnn_cell_16/BiasAdd/ReadVariableOp2T
(simple_rnn_cell_16/MatMul/ReadVariableOp(simple_rnn_cell_16/MatMul/ReadVariableOp2X
*simple_rnn_cell_16/MatMul_1/ReadVariableOp*simple_rnn_cell_16/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :??????????????????
"
_user_specified_name
inputs/0
?C
?
simple_rnn_9_while_body_1954226
2simple_rnn_9_while_simple_rnn_9_while_loop_counter<
8simple_rnn_9_while_simple_rnn_9_while_maximum_iterations"
simple_rnn_9_while_placeholder$
 simple_rnn_9_while_placeholder_1$
 simple_rnn_9_while_placeholder_25
1simple_rnn_9_while_simple_rnn_9_strided_slice_1_0q
msimple_rnn_9_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_9_tensorarrayunstack_tensorlistfromtensor_0J
Fsimple_rnn_9_while_simple_rnn_cell_17_matmul_readvariableop_resource_0K
Gsimple_rnn_9_while_simple_rnn_cell_17_biasadd_readvariableop_resource_0L
Hsimple_rnn_9_while_simple_rnn_cell_17_matmul_1_readvariableop_resource_0
simple_rnn_9_while_identity!
simple_rnn_9_while_identity_1!
simple_rnn_9_while_identity_2!
simple_rnn_9_while_identity_3!
simple_rnn_9_while_identity_43
/simple_rnn_9_while_simple_rnn_9_strided_slice_1o
ksimple_rnn_9_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_9_tensorarrayunstack_tensorlistfromtensorH
Dsimple_rnn_9_while_simple_rnn_cell_17_matmul_readvariableop_resourceI
Esimple_rnn_9_while_simple_rnn_cell_17_biasadd_readvariableop_resourceJ
Fsimple_rnn_9_while_simple_rnn_cell_17_matmul_1_readvariableop_resource??<simple_rnn_9/while/simple_rnn_cell_17/BiasAdd/ReadVariableOp?;simple_rnn_9/while/simple_rnn_cell_17/MatMul/ReadVariableOp?=simple_rnn_9/while/simple_rnn_cell_17/MatMul_1/ReadVariableOp?
Dsimple_rnn_9/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2F
Dsimple_rnn_9/while/TensorArrayV2Read/TensorListGetItem/element_shape?
6simple_rnn_9/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemmsimple_rnn_9_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_9_tensorarrayunstack_tensorlistfromtensor_0simple_rnn_9_while_placeholderMsimple_rnn_9/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype028
6simple_rnn_9/while/TensorArrayV2Read/TensorListGetItem?
;simple_rnn_9/while/simple_rnn_cell_17/MatMul/ReadVariableOpReadVariableOpFsimple_rnn_9_while_simple_rnn_cell_17_matmul_readvariableop_resource_0*
_output_shapes
:	?2*
dtype02=
;simple_rnn_9/while/simple_rnn_cell_17/MatMul/ReadVariableOp?
,simple_rnn_9/while/simple_rnn_cell_17/MatMul	MLCMatMul=simple_rnn_9/while/TensorArrayV2Read/TensorListGetItem:item:0Csimple_rnn_9/while/simple_rnn_cell_17/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22.
,simple_rnn_9/while/simple_rnn_cell_17/MatMul?
<simple_rnn_9/while/simple_rnn_cell_17/BiasAdd/ReadVariableOpReadVariableOpGsimple_rnn_9_while_simple_rnn_cell_17_biasadd_readvariableop_resource_0*
_output_shapes
:2*
dtype02>
<simple_rnn_9/while/simple_rnn_cell_17/BiasAdd/ReadVariableOp?
-simple_rnn_9/while/simple_rnn_cell_17/BiasAddBiasAdd6simple_rnn_9/while/simple_rnn_cell_17/MatMul:product:0Dsimple_rnn_9/while/simple_rnn_cell_17/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22/
-simple_rnn_9/while/simple_rnn_cell_17/BiasAdd?
=simple_rnn_9/while/simple_rnn_cell_17/MatMul_1/ReadVariableOpReadVariableOpHsimple_rnn_9_while_simple_rnn_cell_17_matmul_1_readvariableop_resource_0*
_output_shapes

:22*
dtype02?
=simple_rnn_9/while/simple_rnn_cell_17/MatMul_1/ReadVariableOp?
.simple_rnn_9/while/simple_rnn_cell_17/MatMul_1	MLCMatMul simple_rnn_9_while_placeholder_2Esimple_rnn_9/while/simple_rnn_cell_17/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????220
.simple_rnn_9/while/simple_rnn_cell_17/MatMul_1?
)simple_rnn_9/while/simple_rnn_cell_17/addAddV26simple_rnn_9/while/simple_rnn_cell_17/BiasAdd:output:08simple_rnn_9/while/simple_rnn_cell_17/MatMul_1:product:0*
T0*'
_output_shapes
:?????????22+
)simple_rnn_9/while/simple_rnn_cell_17/add?
*simple_rnn_9/while/simple_rnn_cell_17/TanhTanh-simple_rnn_9/while/simple_rnn_cell_17/add:z:0*
T0*'
_output_shapes
:?????????22,
*simple_rnn_9/while/simple_rnn_cell_17/Tanh?
7simple_rnn_9/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem simple_rnn_9_while_placeholder_1simple_rnn_9_while_placeholder.simple_rnn_9/while/simple_rnn_cell_17/Tanh:y:0*
_output_shapes
: *
element_dtype029
7simple_rnn_9/while/TensorArrayV2Write/TensorListSetItemv
simple_rnn_9/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
simple_rnn_9/while/add/y?
simple_rnn_9/while/addAddV2simple_rnn_9_while_placeholder!simple_rnn_9/while/add/y:output:0*
T0*
_output_shapes
: 2
simple_rnn_9/while/addz
simple_rnn_9/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
simple_rnn_9/while/add_1/y?
simple_rnn_9/while/add_1AddV22simple_rnn_9_while_simple_rnn_9_while_loop_counter#simple_rnn_9/while/add_1/y:output:0*
T0*
_output_shapes
: 2
simple_rnn_9/while/add_1?
simple_rnn_9/while/IdentityIdentitysimple_rnn_9/while/add_1:z:0=^simple_rnn_9/while/simple_rnn_cell_17/BiasAdd/ReadVariableOp<^simple_rnn_9/while/simple_rnn_cell_17/MatMul/ReadVariableOp>^simple_rnn_9/while/simple_rnn_cell_17/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
simple_rnn_9/while/Identity?
simple_rnn_9/while/Identity_1Identity8simple_rnn_9_while_simple_rnn_9_while_maximum_iterations=^simple_rnn_9/while/simple_rnn_cell_17/BiasAdd/ReadVariableOp<^simple_rnn_9/while/simple_rnn_cell_17/MatMul/ReadVariableOp>^simple_rnn_9/while/simple_rnn_cell_17/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
simple_rnn_9/while/Identity_1?
simple_rnn_9/while/Identity_2Identitysimple_rnn_9/while/add:z:0=^simple_rnn_9/while/simple_rnn_cell_17/BiasAdd/ReadVariableOp<^simple_rnn_9/while/simple_rnn_cell_17/MatMul/ReadVariableOp>^simple_rnn_9/while/simple_rnn_cell_17/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
simple_rnn_9/while/Identity_2?
simple_rnn_9/while/Identity_3IdentityGsimple_rnn_9/while/TensorArrayV2Write/TensorListSetItem:output_handle:0=^simple_rnn_9/while/simple_rnn_cell_17/BiasAdd/ReadVariableOp<^simple_rnn_9/while/simple_rnn_cell_17/MatMul/ReadVariableOp>^simple_rnn_9/while/simple_rnn_cell_17/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
simple_rnn_9/while/Identity_3?
simple_rnn_9/while/Identity_4Identity.simple_rnn_9/while/simple_rnn_cell_17/Tanh:y:0=^simple_rnn_9/while/simple_rnn_cell_17/BiasAdd/ReadVariableOp<^simple_rnn_9/while/simple_rnn_cell_17/MatMul/ReadVariableOp>^simple_rnn_9/while/simple_rnn_cell_17/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????22
simple_rnn_9/while/Identity_4"C
simple_rnn_9_while_identity$simple_rnn_9/while/Identity:output:0"G
simple_rnn_9_while_identity_1&simple_rnn_9/while/Identity_1:output:0"G
simple_rnn_9_while_identity_2&simple_rnn_9/while/Identity_2:output:0"G
simple_rnn_9_while_identity_3&simple_rnn_9/while/Identity_3:output:0"G
simple_rnn_9_while_identity_4&simple_rnn_9/while/Identity_4:output:0"d
/simple_rnn_9_while_simple_rnn_9_strided_slice_11simple_rnn_9_while_simple_rnn_9_strided_slice_1_0"?
Esimple_rnn_9_while_simple_rnn_cell_17_biasadd_readvariableop_resourceGsimple_rnn_9_while_simple_rnn_cell_17_biasadd_readvariableop_resource_0"?
Fsimple_rnn_9_while_simple_rnn_cell_17_matmul_1_readvariableop_resourceHsimple_rnn_9_while_simple_rnn_cell_17_matmul_1_readvariableop_resource_0"?
Dsimple_rnn_9_while_simple_rnn_cell_17_matmul_readvariableop_resourceFsimple_rnn_9_while_simple_rnn_cell_17_matmul_readvariableop_resource_0"?
ksimple_rnn_9_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_9_tensorarrayunstack_tensorlistfromtensormsimple_rnn_9_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_9_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :?????????2: : :::2|
<simple_rnn_9/while/simple_rnn_cell_17/BiasAdd/ReadVariableOp<simple_rnn_9/while/simple_rnn_cell_17/BiasAdd/ReadVariableOp2z
;simple_rnn_9/while/simple_rnn_cell_17/MatMul/ReadVariableOp;simple_rnn_9/while/simple_rnn_cell_17/MatMul/ReadVariableOp2~
=simple_rnn_9/while/simple_rnn_cell_17/MatMul_1/ReadVariableOp=simple_rnn_9/while/simple_rnn_cell_17/MatMul_1/ReadVariableOp: 
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
:?????????2:

_output_shapes
: :

_output_shapes
: 
?4
?
while_body_196704
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0=
9while_simple_rnn_cell_17_matmul_readvariableop_resource_0>
:while_simple_rnn_cell_17_biasadd_readvariableop_resource_0?
;while_simple_rnn_cell_17_matmul_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor;
7while_simple_rnn_cell_17_matmul_readvariableop_resource<
8while_simple_rnn_cell_17_biasadd_readvariableop_resource=
9while_simple_rnn_cell_17_matmul_1_readvariableop_resource??/while/simple_rnn_cell_17/BiasAdd/ReadVariableOp?.while/simple_rnn_cell_17/MatMul/ReadVariableOp?0while/simple_rnn_cell_17/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
.while/simple_rnn_cell_17/MatMul/ReadVariableOpReadVariableOp9while_simple_rnn_cell_17_matmul_readvariableop_resource_0*
_output_shapes
:	?2*
dtype020
.while/simple_rnn_cell_17/MatMul/ReadVariableOp?
while/simple_rnn_cell_17/MatMul	MLCMatMul0while/TensorArrayV2Read/TensorListGetItem:item:06while/simple_rnn_cell_17/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22!
while/simple_rnn_cell_17/MatMul?
/while/simple_rnn_cell_17/BiasAdd/ReadVariableOpReadVariableOp:while_simple_rnn_cell_17_biasadd_readvariableop_resource_0*
_output_shapes
:2*
dtype021
/while/simple_rnn_cell_17/BiasAdd/ReadVariableOp?
 while/simple_rnn_cell_17/BiasAddBiasAdd)while/simple_rnn_cell_17/MatMul:product:07while/simple_rnn_cell_17/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22"
 while/simple_rnn_cell_17/BiasAdd?
0while/simple_rnn_cell_17/MatMul_1/ReadVariableOpReadVariableOp;while_simple_rnn_cell_17_matmul_1_readvariableop_resource_0*
_output_shapes

:22*
dtype022
0while/simple_rnn_cell_17/MatMul_1/ReadVariableOp?
!while/simple_rnn_cell_17/MatMul_1	MLCMatMulwhile_placeholder_28while/simple_rnn_cell_17/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22#
!while/simple_rnn_cell_17/MatMul_1?
while/simple_rnn_cell_17/addAddV2)while/simple_rnn_cell_17/BiasAdd:output:0+while/simple_rnn_cell_17/MatMul_1:product:0*
T0*'
_output_shapes
:?????????22
while/simple_rnn_cell_17/add?
while/simple_rnn_cell_17/TanhTanh while/simple_rnn_cell_17/add:z:0*
T0*'
_output_shapes
:?????????22
while/simple_rnn_cell_17/Tanh?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder!while/simple_rnn_cell_17/Tanh:y:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1?
while/IdentityIdentitywhile/add_1:z:00^while/simple_rnn_cell_17/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_17/MatMul/ReadVariableOp1^while/simple_rnn_cell_17/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations0^while/simple_rnn_cell_17/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_17/MatMul/ReadVariableOp1^while/simple_rnn_cell_17/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:00^while/simple_rnn_cell_17/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_17/MatMul/ReadVariableOp1^while/simple_rnn_cell_17/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:00^while/simple_rnn_cell_17/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_17/MatMul/ReadVariableOp1^while/simple_rnn_cell_17/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity!while/simple_rnn_cell_17/Tanh:y:00^while/simple_rnn_cell_17/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_17/MatMul/ReadVariableOp1^while/simple_rnn_cell_17/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????22
while/Identity_4")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"v
8while_simple_rnn_cell_17_biasadd_readvariableop_resource:while_simple_rnn_cell_17_biasadd_readvariableop_resource_0"x
9while_simple_rnn_cell_17_matmul_1_readvariableop_resource;while_simple_rnn_cell_17_matmul_1_readvariableop_resource_0"t
7while_simple_rnn_cell_17_matmul_readvariableop_resource9while_simple_rnn_cell_17_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :?????????2: : :::2b
/while/simple_rnn_cell_17/BiasAdd/ReadVariableOp/while/simple_rnn_cell_17/BiasAdd/ReadVariableOp2`
.while/simple_rnn_cell_17/MatMul/ReadVariableOp.while/simple_rnn_cell_17/MatMul/ReadVariableOp2d
0while/simple_rnn_cell_17/MatMul_1/ReadVariableOp0while/simple_rnn_cell_17/MatMul_1/ReadVariableOp: 
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
:?????????2:

_output_shapes
: :

_output_shapes
: 
?H
?
H__inference_simple_rnn_8_layer_call_and_return_conditional_losses_196244

inputs5
1simple_rnn_cell_16_matmul_readvariableop_resource6
2simple_rnn_cell_16_biasadd_readvariableop_resource7
3simple_rnn_cell_16_matmul_1_readvariableop_resource
identity??)simple_rnn_cell_16/BiasAdd/ReadVariableOp?(simple_rnn_cell_16/MatMul/ReadVariableOp?*simple_rnn_cell_16/MatMul_1/ReadVariableOp?whileD
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
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
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

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
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
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:x?????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
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
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
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
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_2?
(simple_rnn_cell_16/MatMul/ReadVariableOpReadVariableOp1simple_rnn_cell_16_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02*
(simple_rnn_cell_16/MatMul/ReadVariableOp?
simple_rnn_cell_16/MatMul	MLCMatMulstrided_slice_2:output:00simple_rnn_cell_16/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
simple_rnn_cell_16/MatMul?
)simple_rnn_cell_16/BiasAdd/ReadVariableOpReadVariableOp2simple_rnn_cell_16_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)simple_rnn_cell_16/BiasAdd/ReadVariableOp?
simple_rnn_cell_16/BiasAddBiasAdd#simple_rnn_cell_16/MatMul:product:01simple_rnn_cell_16/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
simple_rnn_cell_16/BiasAdd?
*simple_rnn_cell_16/MatMul_1/ReadVariableOpReadVariableOp3simple_rnn_cell_16_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02,
*simple_rnn_cell_16/MatMul_1/ReadVariableOp?
simple_rnn_cell_16/MatMul_1	MLCMatMulzeros:output:02simple_rnn_cell_16/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
simple_rnn_cell_16/MatMul_1?
simple_rnn_cell_16/addAddV2#simple_rnn_cell_16/BiasAdd:output:0%simple_rnn_cell_16/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
simple_rnn_cell_16/add?
simple_rnn_cell_16/TanhTanhsimple_rnn_cell_16/add:z:0*
T0*(
_output_shapes
:??????????2
simple_rnn_cell_16/Tanh?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:01simple_rnn_cell_16_matmul_readvariableop_resource2simple_rnn_cell_16_biasadd_readvariableop_resource3simple_rnn_cell_16_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :??????????: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_196178*
condR
while_cond_196177*9
output_shapes(
&: : : : :??????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:x??????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:?????????x?2
transpose_1?
IdentityIdentitytranspose_1:y:0*^simple_rnn_cell_16/BiasAdd/ReadVariableOp)^simple_rnn_cell_16/MatMul/ReadVariableOp+^simple_rnn_cell_16/MatMul_1/ReadVariableOp^while*
T0*,
_output_shapes
:?????????x?2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????x:::2V
)simple_rnn_cell_16/BiasAdd/ReadVariableOp)simple_rnn_cell_16/BiasAdd/ReadVariableOp2T
(simple_rnn_cell_16/MatMul/ReadVariableOp(simple_rnn_cell_16/MatMul/ReadVariableOp2X
*simple_rnn_cell_16/MatMul_1/ReadVariableOp*simple_rnn_cell_16/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:?????????x
 
_user_specified_nameinputs
?
e
I__inference_activation_21_layer_call_and_return_conditional_losses_195044

inputs
identityN
ReluReluinputs*
T0*'
_output_shapes
:?????????22
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????2:O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs
?
c
E__inference_dropout_9_layer_call_and_return_conditional_losses_196816

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????22

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:?????????22

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:?????????2:O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs
?
?
I__inference_sequential_24_layer_call_and_return_conditional_losses_195107
simple_rnn_8_input
simple_rnn_8_194739
simple_rnn_8_194741
simple_rnn_8_194743
simple_rnn_9_195032
simple_rnn_9_195034
simple_rnn_9_195036
dense_28_195101
dense_28_195103
identity?? dense_28/StatefulPartitionedCall?$simple_rnn_8/StatefulPartitionedCall?$simple_rnn_9/StatefulPartitionedCall?
$simple_rnn_8/StatefulPartitionedCallStatefulPartitionedCallsimple_rnn_8_inputsimple_rnn_8_194739simple_rnn_8_194741simple_rnn_8_194743*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????x?*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_simple_rnn_8_layer_call_and_return_conditional_losses_1946042&
$simple_rnn_8/StatefulPartitionedCall?
activation_20/PartitionedCallPartitionedCall-simple_rnn_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????x?* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_activation_20_layer_call_and_return_conditional_losses_1947512
activation_20/PartitionedCall?
dropout_8/PartitionedCallPartitionedCall&activation_20/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????x?* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_8_layer_call_and_return_conditional_losses_1947682
dropout_8/PartitionedCall?
$simple_rnn_9/StatefulPartitionedCallStatefulPartitionedCall"dropout_8/PartitionedCall:output:0simple_rnn_9_195032simple_rnn_9_195034simple_rnn_9_195036*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_simple_rnn_9_layer_call_and_return_conditional_losses_1948972&
$simple_rnn_9/StatefulPartitionedCall?
activation_21/PartitionedCallPartitionedCall-simple_rnn_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_activation_21_layer_call_and_return_conditional_losses_1950442
activation_21/PartitionedCall?
dropout_9/PartitionedCallPartitionedCall&activation_21/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_9_layer_call_and_return_conditional_losses_1950612
dropout_9/PartitionedCall?
 dense_28/StatefulPartitionedCallStatefulPartitionedCall"dropout_9/PartitionedCall:output:0dense_28_195101dense_28_195103*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????<*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_28_layer_call_and_return_conditional_losses_1950902"
 dense_28/StatefulPartitionedCall?
IdentityIdentity)dense_28/StatefulPartitionedCall:output:0!^dense_28/StatefulPartitionedCall%^simple_rnn_8/StatefulPartitionedCall%^simple_rnn_9/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????<2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:?????????x::::::::2D
 dense_28/StatefulPartitionedCall dense_28/StatefulPartitionedCall2L
$simple_rnn_8/StatefulPartitionedCall$simple_rnn_8/StatefulPartitionedCall2L
$simple_rnn_9/StatefulPartitionedCall$simple_rnn_9/StatefulPartitionedCall:_ [
+
_output_shapes
:?????????x
,
_user_specified_namesimple_rnn_8_input
?
?
while_cond_194537
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_194537___redundant_placeholder04
0while_while_cond_194537___redundant_placeholder14
0while_while_cond_194537___redundant_placeholder24
0while_while_cond_194537___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
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
while_identitywhile/Identity:output:0*A
_input_shapes0
.: : : : :??????????: ::::: 
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
: :.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?H
?
H__inference_simple_rnn_9_layer_call_and_return_conditional_losses_196770
inputs_05
1simple_rnn_cell_17_matmul_readvariableop_resource6
2simple_rnn_cell_17_biasadd_readvariableop_resource7
3simple_rnn_cell_17_matmul_1_readvariableop_resource
identity??)simple_rnn_cell_17/BiasAdd/ReadVariableOp?(simple_rnn_cell_17/MatMul/ReadVariableOp?*simple_rnn_cell_17/MatMul_1/ReadVariableOp?whileF
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
value	B :22
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
value	B :22
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
:?????????22
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*5
_output_shapes#
!:???????????????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
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
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_2?
(simple_rnn_cell_17/MatMul/ReadVariableOpReadVariableOp1simple_rnn_cell_17_matmul_readvariableop_resource*
_output_shapes
:	?2*
dtype02*
(simple_rnn_cell_17/MatMul/ReadVariableOp?
simple_rnn_cell_17/MatMul	MLCMatMulstrided_slice_2:output:00simple_rnn_cell_17/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
simple_rnn_cell_17/MatMul?
)simple_rnn_cell_17/BiasAdd/ReadVariableOpReadVariableOp2simple_rnn_cell_17_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02+
)simple_rnn_cell_17/BiasAdd/ReadVariableOp?
simple_rnn_cell_17/BiasAddBiasAdd#simple_rnn_cell_17/MatMul:product:01simple_rnn_cell_17/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
simple_rnn_cell_17/BiasAdd?
*simple_rnn_cell_17/MatMul_1/ReadVariableOpReadVariableOp3simple_rnn_cell_17_matmul_1_readvariableop_resource*
_output_shapes

:22*
dtype02,
*simple_rnn_cell_17/MatMul_1/ReadVariableOp?
simple_rnn_cell_17/MatMul_1	MLCMatMulzeros:output:02simple_rnn_cell_17/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
simple_rnn_cell_17/MatMul_1?
simple_rnn_cell_17/addAddV2#simple_rnn_cell_17/BiasAdd:output:0%simple_rnn_cell_17/MatMul_1:product:0*
T0*'
_output_shapes
:?????????22
simple_rnn_cell_17/add?
simple_rnn_cell_17/TanhTanhsimple_rnn_cell_17/add:z:0*
T0*'
_output_shapes
:?????????22
simple_rnn_cell_17/Tanh?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:01simple_rnn_cell_17_matmul_readvariableop_resource2simple_rnn_cell_17_biasadd_readvariableop_resource3simple_rnn_cell_17_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????2: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_196704*
condR
while_cond_196703*8
output_shapes'
%: : : : :?????????2: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????2*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????2*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????22
transpose_1?
IdentityIdentitystrided_slice_3:output:0*^simple_rnn_cell_17/BiasAdd/ReadVariableOp)^simple_rnn_cell_17/MatMul/ReadVariableOp+^simple_rnn_cell_17/MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*@
_input_shapes/
-:???????????????????:::2V
)simple_rnn_cell_17/BiasAdd/ReadVariableOp)simple_rnn_cell_17/BiasAdd/ReadVariableOp2T
(simple_rnn_cell_17/MatMul/ReadVariableOp(simple_rnn_cell_17/MatMul/ReadVariableOp2X
*simple_rnn_cell_17/MatMul_1/ReadVariableOp*simple_rnn_cell_17/MatMul_1/ReadVariableOp2
whilewhile:_ [
5
_output_shapes#
!:???????????????????
"
_user_specified_name
inputs/0
?
?
N__inference_simple_rnn_cell_17_layer_call_and_return_conditional_losses_194043

inputs

states"
matmul_readvariableop_resource#
biasadd_readvariableop_resource$
 matmul_1_readvariableop_resource
identity

identity_1??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?MatMul_1/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?2*
dtype02
MatMul/ReadVariableOpv
MatMul	MLCMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:2*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22	
BiasAdd?
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:22*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1	MLCMatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22

MatMul_1k
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*'
_output_shapes
:?????????22
addO
TanhTanhadd:z:0*
T0*'
_output_shapes
:?????????22
Tanh?
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????22

Identity?

Identity_1IdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????22

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*F
_input_shapes5
3:??????????:?????????2:::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????2
 
_user_specified_namestates
?S
?
,sequential_24_simple_rnn_8_while_body_193280R
Nsequential_24_simple_rnn_8_while_sequential_24_simple_rnn_8_while_loop_counterX
Tsequential_24_simple_rnn_8_while_sequential_24_simple_rnn_8_while_maximum_iterations0
,sequential_24_simple_rnn_8_while_placeholder2
.sequential_24_simple_rnn_8_while_placeholder_12
.sequential_24_simple_rnn_8_while_placeholder_2Q
Msequential_24_simple_rnn_8_while_sequential_24_simple_rnn_8_strided_slice_1_0?
?sequential_24_simple_rnn_8_while_tensorarrayv2read_tensorlistgetitem_sequential_24_simple_rnn_8_tensorarrayunstack_tensorlistfromtensor_0X
Tsequential_24_simple_rnn_8_while_simple_rnn_cell_16_matmul_readvariableop_resource_0Y
Usequential_24_simple_rnn_8_while_simple_rnn_cell_16_biasadd_readvariableop_resource_0Z
Vsequential_24_simple_rnn_8_while_simple_rnn_cell_16_matmul_1_readvariableop_resource_0-
)sequential_24_simple_rnn_8_while_identity/
+sequential_24_simple_rnn_8_while_identity_1/
+sequential_24_simple_rnn_8_while_identity_2/
+sequential_24_simple_rnn_8_while_identity_3/
+sequential_24_simple_rnn_8_while_identity_4O
Ksequential_24_simple_rnn_8_while_sequential_24_simple_rnn_8_strided_slice_1?
?sequential_24_simple_rnn_8_while_tensorarrayv2read_tensorlistgetitem_sequential_24_simple_rnn_8_tensorarrayunstack_tensorlistfromtensorV
Rsequential_24_simple_rnn_8_while_simple_rnn_cell_16_matmul_readvariableop_resourceW
Ssequential_24_simple_rnn_8_while_simple_rnn_cell_16_biasadd_readvariableop_resourceX
Tsequential_24_simple_rnn_8_while_simple_rnn_cell_16_matmul_1_readvariableop_resource??Jsequential_24/simple_rnn_8/while/simple_rnn_cell_16/BiasAdd/ReadVariableOp?Isequential_24/simple_rnn_8/while/simple_rnn_cell_16/MatMul/ReadVariableOp?Ksequential_24/simple_rnn_8/while/simple_rnn_cell_16/MatMul_1/ReadVariableOp?
Rsequential_24/simple_rnn_8/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2T
Rsequential_24/simple_rnn_8/while/TensorArrayV2Read/TensorListGetItem/element_shape?
Dsequential_24/simple_rnn_8/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem?sequential_24_simple_rnn_8_while_tensorarrayv2read_tensorlistgetitem_sequential_24_simple_rnn_8_tensorarrayunstack_tensorlistfromtensor_0,sequential_24_simple_rnn_8_while_placeholder[sequential_24/simple_rnn_8/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02F
Dsequential_24/simple_rnn_8/while/TensorArrayV2Read/TensorListGetItem?
Isequential_24/simple_rnn_8/while/simple_rnn_cell_16/MatMul/ReadVariableOpReadVariableOpTsequential_24_simple_rnn_8_while_simple_rnn_cell_16_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype02K
Isequential_24/simple_rnn_8/while/simple_rnn_cell_16/MatMul/ReadVariableOp?
:sequential_24/simple_rnn_8/while/simple_rnn_cell_16/MatMul	MLCMatMulKsequential_24/simple_rnn_8/while/TensorArrayV2Read/TensorListGetItem:item:0Qsequential_24/simple_rnn_8/while/simple_rnn_cell_16/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2<
:sequential_24/simple_rnn_8/while/simple_rnn_cell_16/MatMul?
Jsequential_24/simple_rnn_8/while/simple_rnn_cell_16/BiasAdd/ReadVariableOpReadVariableOpUsequential_24_simple_rnn_8_while_simple_rnn_cell_16_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02L
Jsequential_24/simple_rnn_8/while/simple_rnn_cell_16/BiasAdd/ReadVariableOp?
;sequential_24/simple_rnn_8/while/simple_rnn_cell_16/BiasAddBiasAddDsequential_24/simple_rnn_8/while/simple_rnn_cell_16/MatMul:product:0Rsequential_24/simple_rnn_8/while/simple_rnn_cell_16/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2=
;sequential_24/simple_rnn_8/while/simple_rnn_cell_16/BiasAdd?
Ksequential_24/simple_rnn_8/while/simple_rnn_cell_16/MatMul_1/ReadVariableOpReadVariableOpVsequential_24_simple_rnn_8_while_simple_rnn_cell_16_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02M
Ksequential_24/simple_rnn_8/while/simple_rnn_cell_16/MatMul_1/ReadVariableOp?
<sequential_24/simple_rnn_8/while/simple_rnn_cell_16/MatMul_1	MLCMatMul.sequential_24_simple_rnn_8_while_placeholder_2Ssequential_24/simple_rnn_8/while/simple_rnn_cell_16/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2>
<sequential_24/simple_rnn_8/while/simple_rnn_cell_16/MatMul_1?
7sequential_24/simple_rnn_8/while/simple_rnn_cell_16/addAddV2Dsequential_24/simple_rnn_8/while/simple_rnn_cell_16/BiasAdd:output:0Fsequential_24/simple_rnn_8/while/simple_rnn_cell_16/MatMul_1:product:0*
T0*(
_output_shapes
:??????????29
7sequential_24/simple_rnn_8/while/simple_rnn_cell_16/add?
8sequential_24/simple_rnn_8/while/simple_rnn_cell_16/TanhTanh;sequential_24/simple_rnn_8/while/simple_rnn_cell_16/add:z:0*
T0*(
_output_shapes
:??????????2:
8sequential_24/simple_rnn_8/while/simple_rnn_cell_16/Tanh?
Esequential_24/simple_rnn_8/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem.sequential_24_simple_rnn_8_while_placeholder_1,sequential_24_simple_rnn_8_while_placeholder<sequential_24/simple_rnn_8/while/simple_rnn_cell_16/Tanh:y:0*
_output_shapes
: *
element_dtype02G
Esequential_24/simple_rnn_8/while/TensorArrayV2Write/TensorListSetItem?
&sequential_24/simple_rnn_8/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2(
&sequential_24/simple_rnn_8/while/add/y?
$sequential_24/simple_rnn_8/while/addAddV2,sequential_24_simple_rnn_8_while_placeholder/sequential_24/simple_rnn_8/while/add/y:output:0*
T0*
_output_shapes
: 2&
$sequential_24/simple_rnn_8/while/add?
(sequential_24/simple_rnn_8/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2*
(sequential_24/simple_rnn_8/while/add_1/y?
&sequential_24/simple_rnn_8/while/add_1AddV2Nsequential_24_simple_rnn_8_while_sequential_24_simple_rnn_8_while_loop_counter1sequential_24/simple_rnn_8/while/add_1/y:output:0*
T0*
_output_shapes
: 2(
&sequential_24/simple_rnn_8/while/add_1?
)sequential_24/simple_rnn_8/while/IdentityIdentity*sequential_24/simple_rnn_8/while/add_1:z:0K^sequential_24/simple_rnn_8/while/simple_rnn_cell_16/BiasAdd/ReadVariableOpJ^sequential_24/simple_rnn_8/while/simple_rnn_cell_16/MatMul/ReadVariableOpL^sequential_24/simple_rnn_8/while/simple_rnn_cell_16/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2+
)sequential_24/simple_rnn_8/while/Identity?
+sequential_24/simple_rnn_8/while/Identity_1IdentityTsequential_24_simple_rnn_8_while_sequential_24_simple_rnn_8_while_maximum_iterationsK^sequential_24/simple_rnn_8/while/simple_rnn_cell_16/BiasAdd/ReadVariableOpJ^sequential_24/simple_rnn_8/while/simple_rnn_cell_16/MatMul/ReadVariableOpL^sequential_24/simple_rnn_8/while/simple_rnn_cell_16/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2-
+sequential_24/simple_rnn_8/while/Identity_1?
+sequential_24/simple_rnn_8/while/Identity_2Identity(sequential_24/simple_rnn_8/while/add:z:0K^sequential_24/simple_rnn_8/while/simple_rnn_cell_16/BiasAdd/ReadVariableOpJ^sequential_24/simple_rnn_8/while/simple_rnn_cell_16/MatMul/ReadVariableOpL^sequential_24/simple_rnn_8/while/simple_rnn_cell_16/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2-
+sequential_24/simple_rnn_8/while/Identity_2?
+sequential_24/simple_rnn_8/while/Identity_3IdentityUsequential_24/simple_rnn_8/while/TensorArrayV2Write/TensorListSetItem:output_handle:0K^sequential_24/simple_rnn_8/while/simple_rnn_cell_16/BiasAdd/ReadVariableOpJ^sequential_24/simple_rnn_8/while/simple_rnn_cell_16/MatMul/ReadVariableOpL^sequential_24/simple_rnn_8/while/simple_rnn_cell_16/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2-
+sequential_24/simple_rnn_8/while/Identity_3?
+sequential_24/simple_rnn_8/while/Identity_4Identity<sequential_24/simple_rnn_8/while/simple_rnn_cell_16/Tanh:y:0K^sequential_24/simple_rnn_8/while/simple_rnn_cell_16/BiasAdd/ReadVariableOpJ^sequential_24/simple_rnn_8/while/simple_rnn_cell_16/MatMul/ReadVariableOpL^sequential_24/simple_rnn_8/while/simple_rnn_cell_16/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2-
+sequential_24/simple_rnn_8/while/Identity_4"_
)sequential_24_simple_rnn_8_while_identity2sequential_24/simple_rnn_8/while/Identity:output:0"c
+sequential_24_simple_rnn_8_while_identity_14sequential_24/simple_rnn_8/while/Identity_1:output:0"c
+sequential_24_simple_rnn_8_while_identity_24sequential_24/simple_rnn_8/while/Identity_2:output:0"c
+sequential_24_simple_rnn_8_while_identity_34sequential_24/simple_rnn_8/while/Identity_3:output:0"c
+sequential_24_simple_rnn_8_while_identity_44sequential_24/simple_rnn_8/while/Identity_4:output:0"?
Ksequential_24_simple_rnn_8_while_sequential_24_simple_rnn_8_strided_slice_1Msequential_24_simple_rnn_8_while_sequential_24_simple_rnn_8_strided_slice_1_0"?
Ssequential_24_simple_rnn_8_while_simple_rnn_cell_16_biasadd_readvariableop_resourceUsequential_24_simple_rnn_8_while_simple_rnn_cell_16_biasadd_readvariableop_resource_0"?
Tsequential_24_simple_rnn_8_while_simple_rnn_cell_16_matmul_1_readvariableop_resourceVsequential_24_simple_rnn_8_while_simple_rnn_cell_16_matmul_1_readvariableop_resource_0"?
Rsequential_24_simple_rnn_8_while_simple_rnn_cell_16_matmul_readvariableop_resourceTsequential_24_simple_rnn_8_while_simple_rnn_cell_16_matmul_readvariableop_resource_0"?
?sequential_24_simple_rnn_8_while_tensorarrayv2read_tensorlistgetitem_sequential_24_simple_rnn_8_tensorarrayunstack_tensorlistfromtensor?sequential_24_simple_rnn_8_while_tensorarrayv2read_tensorlistgetitem_sequential_24_simple_rnn_8_tensorarrayunstack_tensorlistfromtensor_0*?
_input_shapes.
,: : : : :??????????: : :::2?
Jsequential_24/simple_rnn_8/while/simple_rnn_cell_16/BiasAdd/ReadVariableOpJsequential_24/simple_rnn_8/while/simple_rnn_cell_16/BiasAdd/ReadVariableOp2?
Isequential_24/simple_rnn_8/while/simple_rnn_cell_16/MatMul/ReadVariableOpIsequential_24/simple_rnn_8/while/simple_rnn_cell_16/MatMul/ReadVariableOp2?
Ksequential_24/simple_rnn_8/while/simple_rnn_cell_16/MatMul_1/ReadVariableOpKsequential_24/simple_rnn_8/while/simple_rnn_cell_16/MatMul_1/ReadVariableOp: 
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
: :.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?
F
*__inference_dropout_9_layer_call_fn_196826

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_9_layer_call_and_return_conditional_losses_1950662
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????2:O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs
?
F
*__inference_dropout_8_layer_call_fn_196300

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????x?* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_8_layer_call_and_return_conditional_losses_1947732
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:?????????x?2

Identity"
identityIdentity:output:0*+
_input_shapes
:?????????x?:T P
,
_output_shapes
:?????????x?
 
_user_specified_nameinputs
?4
?
while_body_196066
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0=
9while_simple_rnn_cell_16_matmul_readvariableop_resource_0>
:while_simple_rnn_cell_16_biasadd_readvariableop_resource_0?
;while_simple_rnn_cell_16_matmul_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor;
7while_simple_rnn_cell_16_matmul_readvariableop_resource<
8while_simple_rnn_cell_16_biasadd_readvariableop_resource=
9while_simple_rnn_cell_16_matmul_1_readvariableop_resource??/while/simple_rnn_cell_16/BiasAdd/ReadVariableOp?.while/simple_rnn_cell_16/MatMul/ReadVariableOp?0while/simple_rnn_cell_16/MatMul_1/ReadVariableOp?
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
.while/simple_rnn_cell_16/MatMul/ReadVariableOpReadVariableOp9while_simple_rnn_cell_16_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype020
.while/simple_rnn_cell_16/MatMul/ReadVariableOp?
while/simple_rnn_cell_16/MatMul	MLCMatMul0while/TensorArrayV2Read/TensorListGetItem:item:06while/simple_rnn_cell_16/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2!
while/simple_rnn_cell_16/MatMul?
/while/simple_rnn_cell_16/BiasAdd/ReadVariableOpReadVariableOp:while_simple_rnn_cell_16_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype021
/while/simple_rnn_cell_16/BiasAdd/ReadVariableOp?
 while/simple_rnn_cell_16/BiasAddBiasAdd)while/simple_rnn_cell_16/MatMul:product:07while/simple_rnn_cell_16/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2"
 while/simple_rnn_cell_16/BiasAdd?
0while/simple_rnn_cell_16/MatMul_1/ReadVariableOpReadVariableOp;while_simple_rnn_cell_16_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype022
0while/simple_rnn_cell_16/MatMul_1/ReadVariableOp?
!while/simple_rnn_cell_16/MatMul_1	MLCMatMulwhile_placeholder_28while/simple_rnn_cell_16/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2#
!while/simple_rnn_cell_16/MatMul_1?
while/simple_rnn_cell_16/addAddV2)while/simple_rnn_cell_16/BiasAdd:output:0+while/simple_rnn_cell_16/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
while/simple_rnn_cell_16/add?
while/simple_rnn_cell_16/TanhTanh while/simple_rnn_cell_16/add:z:0*
T0*(
_output_shapes
:??????????2
while/simple_rnn_cell_16/Tanh?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder!while/simple_rnn_cell_16/Tanh:y:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1?
while/IdentityIdentitywhile/add_1:z:00^while/simple_rnn_cell_16/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_16/MatMul/ReadVariableOp1^while/simple_rnn_cell_16/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations0^while/simple_rnn_cell_16/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_16/MatMul/ReadVariableOp1^while/simple_rnn_cell_16/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:00^while/simple_rnn_cell_16/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_16/MatMul/ReadVariableOp1^while/simple_rnn_cell_16/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:00^while/simple_rnn_cell_16/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_16/MatMul/ReadVariableOp1^while/simple_rnn_cell_16/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity!while/simple_rnn_cell_16/Tanh:y:00^while/simple_rnn_cell_16/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_16/MatMul/ReadVariableOp1^while/simple_rnn_cell_16/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2
while/Identity_4")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"v
8while_simple_rnn_cell_16_biasadd_readvariableop_resource:while_simple_rnn_cell_16_biasadd_readvariableop_resource_0"x
9while_simple_rnn_cell_16_matmul_1_readvariableop_resource;while_simple_rnn_cell_16_matmul_1_readvariableop_resource_0"t
7while_simple_rnn_cell_16_matmul_readvariableop_resource9while_simple_rnn_cell_16_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*?
_input_shapes.
,: : : : :??????????: : :::2b
/while/simple_rnn_cell_16/BiasAdd/ReadVariableOp/while/simple_rnn_cell_16/BiasAdd/ReadVariableOp2`
.while/simple_rnn_cell_16/MatMul/ReadVariableOp.while/simple_rnn_cell_16/MatMul/ReadVariableOp2d
0while/simple_rnn_cell_16/MatMul_1/ReadVariableOp0while/simple_rnn_cell_16/MatMul_1/ReadVariableOp: 
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
: :.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?<
?
H__inference_simple_rnn_9_layer_call_and_return_conditional_losses_194480

inputs
simple_rnn_cell_17_194405
simple_rnn_cell_17_194407
simple_rnn_cell_17_194409
identity??*simple_rnn_cell_17/StatefulPartitionedCall?whileD
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
value	B :22
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
value	B :22
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
:?????????22
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:???????????????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
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
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_2?
*simple_rnn_cell_17/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0simple_rnn_cell_17_194405simple_rnn_cell_17_194407simple_rnn_cell_17_194409*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????2:?????????2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_simple_rnn_cell_17_layer_call_and_return_conditional_losses_1940432,
*simple_rnn_cell_17/StatefulPartitionedCall?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
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
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0simple_rnn_cell_17_194405simple_rnn_cell_17_194407simple_rnn_cell_17_194409*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????2: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_194417*
condR
while_cond_194416*8
output_shapes'
%: : : : :?????????2: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????2*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????2*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????22
transpose_1?
IdentityIdentitystrided_slice_3:output:0+^simple_rnn_cell_17/StatefulPartitionedCall^while*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*@
_input_shapes/
-:???????????????????:::2X
*simple_rnn_cell_17/StatefulPartitionedCall*simple_rnn_cell_17/StatefulPartitionedCall2
whilewhile:] Y
5
_output_shapes#
!:???????????????????
 
_user_specified_nameinputs
?	
?
3__inference_simple_rnn_cell_16_layer_call_fn_196894

inputs
states_0
unknown
	unknown_0
	unknown_1
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:??????????:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_simple_rnn_cell_16_layer_call_and_return_conditional_losses_1935142
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*F
_input_shapes5
3:?????????:??????????:::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/0
?4
?
while_body_196178
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0=
9while_simple_rnn_cell_16_matmul_readvariableop_resource_0>
:while_simple_rnn_cell_16_biasadd_readvariableop_resource_0?
;while_simple_rnn_cell_16_matmul_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor;
7while_simple_rnn_cell_16_matmul_readvariableop_resource<
8while_simple_rnn_cell_16_biasadd_readvariableop_resource=
9while_simple_rnn_cell_16_matmul_1_readvariableop_resource??/while/simple_rnn_cell_16/BiasAdd/ReadVariableOp?.while/simple_rnn_cell_16/MatMul/ReadVariableOp?0while/simple_rnn_cell_16/MatMul_1/ReadVariableOp?
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
.while/simple_rnn_cell_16/MatMul/ReadVariableOpReadVariableOp9while_simple_rnn_cell_16_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype020
.while/simple_rnn_cell_16/MatMul/ReadVariableOp?
while/simple_rnn_cell_16/MatMul	MLCMatMul0while/TensorArrayV2Read/TensorListGetItem:item:06while/simple_rnn_cell_16/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2!
while/simple_rnn_cell_16/MatMul?
/while/simple_rnn_cell_16/BiasAdd/ReadVariableOpReadVariableOp:while_simple_rnn_cell_16_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype021
/while/simple_rnn_cell_16/BiasAdd/ReadVariableOp?
 while/simple_rnn_cell_16/BiasAddBiasAdd)while/simple_rnn_cell_16/MatMul:product:07while/simple_rnn_cell_16/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2"
 while/simple_rnn_cell_16/BiasAdd?
0while/simple_rnn_cell_16/MatMul_1/ReadVariableOpReadVariableOp;while_simple_rnn_cell_16_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype022
0while/simple_rnn_cell_16/MatMul_1/ReadVariableOp?
!while/simple_rnn_cell_16/MatMul_1	MLCMatMulwhile_placeholder_28while/simple_rnn_cell_16/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2#
!while/simple_rnn_cell_16/MatMul_1?
while/simple_rnn_cell_16/addAddV2)while/simple_rnn_cell_16/BiasAdd:output:0+while/simple_rnn_cell_16/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
while/simple_rnn_cell_16/add?
while/simple_rnn_cell_16/TanhTanh while/simple_rnn_cell_16/add:z:0*
T0*(
_output_shapes
:??????????2
while/simple_rnn_cell_16/Tanh?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder!while/simple_rnn_cell_16/Tanh:y:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1?
while/IdentityIdentitywhile/add_1:z:00^while/simple_rnn_cell_16/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_16/MatMul/ReadVariableOp1^while/simple_rnn_cell_16/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations0^while/simple_rnn_cell_16/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_16/MatMul/ReadVariableOp1^while/simple_rnn_cell_16/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:00^while/simple_rnn_cell_16/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_16/MatMul/ReadVariableOp1^while/simple_rnn_cell_16/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:00^while/simple_rnn_cell_16/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_16/MatMul/ReadVariableOp1^while/simple_rnn_cell_16/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity!while/simple_rnn_cell_16/Tanh:y:00^while/simple_rnn_cell_16/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_16/MatMul/ReadVariableOp1^while/simple_rnn_cell_16/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2
while/Identity_4")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"v
8while_simple_rnn_cell_16_biasadd_readvariableop_resource:while_simple_rnn_cell_16_biasadd_readvariableop_resource_0"x
9while_simple_rnn_cell_16_matmul_1_readvariableop_resource;while_simple_rnn_cell_16_matmul_1_readvariableop_resource_0"t
7while_simple_rnn_cell_16_matmul_readvariableop_resource9while_simple_rnn_cell_16_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*?
_input_shapes.
,: : : : :??????????: : :::2b
/while/simple_rnn_cell_16/BiasAdd/ReadVariableOp/while/simple_rnn_cell_16/BiasAdd/ReadVariableOp2`
.while/simple_rnn_cell_16/MatMul/ReadVariableOp.while/simple_rnn_cell_16/MatMul/ReadVariableOp2d
0while/simple_rnn_cell_16/MatMul_1/ReadVariableOp0while/simple_rnn_cell_16/MatMul_1/ReadVariableOp: 
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
: :.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?
?
while_cond_194649
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_194649___redundant_placeholder04
0while_while_cond_194649___redundant_placeholder14
0while_while_cond_194649___redundant_placeholder24
0while_while_cond_194649___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
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
while_identitywhile/Identity:output:0*A
_input_shapes0
.: : : : :??????????: ::::: 
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
: :.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?
?
while_cond_194299
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_194299___redundant_placeholder04
0while_while_cond_194299___redundant_placeholder14
0while_while_cond_194299___redundant_placeholder24
0while_while_cond_194299___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
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
while_identitywhile/Identity:output:0*@
_input_shapes/
-: : : : :?????????2: ::::: 
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
:?????????2:

_output_shapes
: :

_output_shapes
:
?O
?
__inference__traced_save_197098
file_prefix.
*savev2_dense_28_kernel_read_readvariableop,
(savev2_dense_28_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableopE
Asavev2_simple_rnn_8_simple_rnn_cell_16_kernel_read_readvariableopO
Ksavev2_simple_rnn_8_simple_rnn_cell_16_recurrent_kernel_read_readvariableopC
?savev2_simple_rnn_8_simple_rnn_cell_16_bias_read_readvariableopE
Asavev2_simple_rnn_9_simple_rnn_cell_17_kernel_read_readvariableopO
Ksavev2_simple_rnn_9_simple_rnn_cell_17_recurrent_kernel_read_readvariableopC
?savev2_simple_rnn_9_simple_rnn_cell_17_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop&
"savev2_total_2_read_readvariableop&
"savev2_count_2_read_readvariableop5
1savev2_adam_dense_28_kernel_m_read_readvariableop3
/savev2_adam_dense_28_bias_m_read_readvariableopL
Hsavev2_adam_simple_rnn_8_simple_rnn_cell_16_kernel_m_read_readvariableopV
Rsavev2_adam_simple_rnn_8_simple_rnn_cell_16_recurrent_kernel_m_read_readvariableopJ
Fsavev2_adam_simple_rnn_8_simple_rnn_cell_16_bias_m_read_readvariableopL
Hsavev2_adam_simple_rnn_9_simple_rnn_cell_17_kernel_m_read_readvariableopV
Rsavev2_adam_simple_rnn_9_simple_rnn_cell_17_recurrent_kernel_m_read_readvariableopJ
Fsavev2_adam_simple_rnn_9_simple_rnn_cell_17_bias_m_read_readvariableop5
1savev2_adam_dense_28_kernel_v_read_readvariableop3
/savev2_adam_dense_28_bias_v_read_readvariableopL
Hsavev2_adam_simple_rnn_8_simple_rnn_cell_16_kernel_v_read_readvariableopV
Rsavev2_adam_simple_rnn_8_simple_rnn_cell_16_recurrent_kernel_v_read_readvariableopJ
Fsavev2_adam_simple_rnn_8_simple_rnn_cell_16_bias_v_read_readvariableopL
Hsavev2_adam_simple_rnn_9_simple_rnn_cell_17_kernel_v_read_readvariableopV
Rsavev2_adam_simple_rnn_9_simple_rnn_cell_17_recurrent_kernel_v_read_readvariableopJ
Fsavev2_adam_simple_rnn_9_simple_rnn_cell_17_bias_v_read_readvariableop
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
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:$*
dtype0*?
value?B?$B6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:$*
dtype0*[
valueRBP$B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_28_kernel_read_readvariableop(savev2_dense_28_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableopAsavev2_simple_rnn_8_simple_rnn_cell_16_kernel_read_readvariableopKsavev2_simple_rnn_8_simple_rnn_cell_16_recurrent_kernel_read_readvariableop?savev2_simple_rnn_8_simple_rnn_cell_16_bias_read_readvariableopAsavev2_simple_rnn_9_simple_rnn_cell_17_kernel_read_readvariableopKsavev2_simple_rnn_9_simple_rnn_cell_17_recurrent_kernel_read_readvariableop?savev2_simple_rnn_9_simple_rnn_cell_17_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop"savev2_total_2_read_readvariableop"savev2_count_2_read_readvariableop1savev2_adam_dense_28_kernel_m_read_readvariableop/savev2_adam_dense_28_bias_m_read_readvariableopHsavev2_adam_simple_rnn_8_simple_rnn_cell_16_kernel_m_read_readvariableopRsavev2_adam_simple_rnn_8_simple_rnn_cell_16_recurrent_kernel_m_read_readvariableopFsavev2_adam_simple_rnn_8_simple_rnn_cell_16_bias_m_read_readvariableopHsavev2_adam_simple_rnn_9_simple_rnn_cell_17_kernel_m_read_readvariableopRsavev2_adam_simple_rnn_9_simple_rnn_cell_17_recurrent_kernel_m_read_readvariableopFsavev2_adam_simple_rnn_9_simple_rnn_cell_17_bias_m_read_readvariableop1savev2_adam_dense_28_kernel_v_read_readvariableop/savev2_adam_dense_28_bias_v_read_readvariableopHsavev2_adam_simple_rnn_8_simple_rnn_cell_16_kernel_v_read_readvariableopRsavev2_adam_simple_rnn_8_simple_rnn_cell_16_recurrent_kernel_v_read_readvariableopFsavev2_adam_simple_rnn_8_simple_rnn_cell_16_bias_v_read_readvariableopHsavev2_adam_simple_rnn_9_simple_rnn_cell_17_kernel_v_read_readvariableopRsavev2_adam_simple_rnn_9_simple_rnn_cell_17_recurrent_kernel_v_read_readvariableopFsavev2_adam_simple_rnn_9_simple_rnn_cell_17_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *2
dtypes(
&2$	2
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

identity_1Identity_1:output:0*?
_input_shapes?
?: :2<:<: : : : : :	?:
??:?:	?2:22:2: : : : : : :2<:<:	?:
??:?:	?2:22:2:2<:<:	?:
??:?:	?2:22:2: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:2<: 

_output_shapes
:<:
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
: :%!

_output_shapes
:	?:&	"
 
_output_shapes
:
??:!


_output_shapes	
:?:%!

_output_shapes
:	?2:$ 

_output_shapes

:22: 

_output_shapes
:2:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:2<: 

_output_shapes
:<:%!

_output_shapes
:	?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:%!

_output_shapes
:	?2:$ 

_output_shapes

:22: 

_output_shapes
:2:$ 

_output_shapes

:2<: 

_output_shapes
:<:%!

_output_shapes
:	?:&"
 
_output_shapes
:
??:! 

_output_shapes	
:?:%!!

_output_shapes
:	?2:$" 

_output_shapes

:22: #

_output_shapes
:2:$

_output_shapes
: 
?
?
N__inference_simple_rnn_cell_16_layer_call_and_return_conditional_losses_193514

inputs

states"
matmul_readvariableop_resource#
biasadd_readvariableop_resource$
 matmul_1_readvariableop_resource
identity

identity_1??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?MatMul_1/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOpw
MatMul	MLCMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAdd?
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul_1/ReadVariableOp}
MatMul_1	MLCMatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2

MatMul_1l
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
addP
TanhTanhadd:z:0*
T0*(
_output_shapes
:??????????2
Tanh?
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity?

Identity_1IdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*F
_input_shapes5
3:?????????:??????????:::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:PL
(
_output_shapes
:??????????
 
_user_specified_namestates
?

?
D__inference_dense_28_layer_call_and_return_conditional_losses_196837

inputs%
!mlcmatmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MLCMatMul/ReadVariableOp?
MLCMatMul/ReadVariableOpReadVariableOp!mlcmatmul_readvariableop_resource*
_output_shapes

:2<*
dtype02
MLCMatMul/ReadVariableOp
	MLCMatMul	MLCMatMulinputs MLCMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2
	MLCMatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:<*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMLCMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<2	
BiasAddX
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:?????????<2
Tanh?
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MLCMatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????<2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????2::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
MLCMatMul/ReadVariableOpMLCMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs
?4
?
while_body_196592
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0=
9while_simple_rnn_cell_17_matmul_readvariableop_resource_0>
:while_simple_rnn_cell_17_biasadd_readvariableop_resource_0?
;while_simple_rnn_cell_17_matmul_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor;
7while_simple_rnn_cell_17_matmul_readvariableop_resource<
8while_simple_rnn_cell_17_biasadd_readvariableop_resource=
9while_simple_rnn_cell_17_matmul_1_readvariableop_resource??/while/simple_rnn_cell_17/BiasAdd/ReadVariableOp?.while/simple_rnn_cell_17/MatMul/ReadVariableOp?0while/simple_rnn_cell_17/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
.while/simple_rnn_cell_17/MatMul/ReadVariableOpReadVariableOp9while_simple_rnn_cell_17_matmul_readvariableop_resource_0*
_output_shapes
:	?2*
dtype020
.while/simple_rnn_cell_17/MatMul/ReadVariableOp?
while/simple_rnn_cell_17/MatMul	MLCMatMul0while/TensorArrayV2Read/TensorListGetItem:item:06while/simple_rnn_cell_17/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22!
while/simple_rnn_cell_17/MatMul?
/while/simple_rnn_cell_17/BiasAdd/ReadVariableOpReadVariableOp:while_simple_rnn_cell_17_biasadd_readvariableop_resource_0*
_output_shapes
:2*
dtype021
/while/simple_rnn_cell_17/BiasAdd/ReadVariableOp?
 while/simple_rnn_cell_17/BiasAddBiasAdd)while/simple_rnn_cell_17/MatMul:product:07while/simple_rnn_cell_17/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22"
 while/simple_rnn_cell_17/BiasAdd?
0while/simple_rnn_cell_17/MatMul_1/ReadVariableOpReadVariableOp;while_simple_rnn_cell_17_matmul_1_readvariableop_resource_0*
_output_shapes

:22*
dtype022
0while/simple_rnn_cell_17/MatMul_1/ReadVariableOp?
!while/simple_rnn_cell_17/MatMul_1	MLCMatMulwhile_placeholder_28while/simple_rnn_cell_17/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22#
!while/simple_rnn_cell_17/MatMul_1?
while/simple_rnn_cell_17/addAddV2)while/simple_rnn_cell_17/BiasAdd:output:0+while/simple_rnn_cell_17/MatMul_1:product:0*
T0*'
_output_shapes
:?????????22
while/simple_rnn_cell_17/add?
while/simple_rnn_cell_17/TanhTanh while/simple_rnn_cell_17/add:z:0*
T0*'
_output_shapes
:?????????22
while/simple_rnn_cell_17/Tanh?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder!while/simple_rnn_cell_17/Tanh:y:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1?
while/IdentityIdentitywhile/add_1:z:00^while/simple_rnn_cell_17/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_17/MatMul/ReadVariableOp1^while/simple_rnn_cell_17/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations0^while/simple_rnn_cell_17/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_17/MatMul/ReadVariableOp1^while/simple_rnn_cell_17/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:00^while/simple_rnn_cell_17/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_17/MatMul/ReadVariableOp1^while/simple_rnn_cell_17/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:00^while/simple_rnn_cell_17/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_17/MatMul/ReadVariableOp1^while/simple_rnn_cell_17/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity!while/simple_rnn_cell_17/Tanh:y:00^while/simple_rnn_cell_17/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_17/MatMul/ReadVariableOp1^while/simple_rnn_cell_17/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????22
while/Identity_4")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"v
8while_simple_rnn_cell_17_biasadd_readvariableop_resource:while_simple_rnn_cell_17_biasadd_readvariableop_resource_0"x
9while_simple_rnn_cell_17_matmul_1_readvariableop_resource;while_simple_rnn_cell_17_matmul_1_readvariableop_resource_0"t
7while_simple_rnn_cell_17_matmul_readvariableop_resource9while_simple_rnn_cell_17_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :?????????2: : :::2b
/while/simple_rnn_cell_17/BiasAdd/ReadVariableOp/while/simple_rnn_cell_17/BiasAdd/ReadVariableOp2`
.while/simple_rnn_cell_17/MatMul/ReadVariableOp.while/simple_rnn_cell_17/MatMul/ReadVariableOp2d
0while/simple_rnn_cell_17/MatMul_1/ReadVariableOp0while/simple_rnn_cell_17/MatMul_1/ReadVariableOp: 
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
:?????????2:

_output_shapes
: :

_output_shapes
: 
?=
?
H__inference_simple_rnn_8_layer_call_and_return_conditional_losses_193851

inputs
simple_rnn_cell_16_193776
simple_rnn_cell_16_193778
simple_rnn_cell_16_193780
identity??*simple_rnn_cell_16/StatefulPartitionedCall?whileD
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
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
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

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
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
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
zerosu
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
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
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
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
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
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_2?
*simple_rnn_cell_16/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0simple_rnn_cell_16_193776simple_rnn_cell_16_193778simple_rnn_cell_16_193780*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:??????????:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_simple_rnn_cell_16_layer_call_and_return_conditional_losses_1935142,
*simple_rnn_cell_16/StatefulPartitionedCall?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
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
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0simple_rnn_cell_16_193776simple_rnn_cell_16_193778simple_rnn_cell_16_193780*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :??????????: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_193788*
condR
while_cond_193787*9
output_shapes(
&: : : : :??????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:???????????????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:???????????????????2
transpose_1?
IdentityIdentitytranspose_1:y:0+^simple_rnn_cell_16/StatefulPartitionedCall^while*
T0*5
_output_shapes#
!:???????????????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????:::2X
*simple_rnn_cell_16/StatefulPartitionedCall*simple_rnn_cell_16/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
?
I__inference_sequential_24_layer_call_and_return_conditional_losses_195134
simple_rnn_8_input
simple_rnn_8_195110
simple_rnn_8_195112
simple_rnn_8_195114
simple_rnn_9_195119
simple_rnn_9_195121
simple_rnn_9_195123
dense_28_195128
dense_28_195130
identity?? dense_28/StatefulPartitionedCall?$simple_rnn_8/StatefulPartitionedCall?$simple_rnn_9/StatefulPartitionedCall?
$simple_rnn_8/StatefulPartitionedCallStatefulPartitionedCallsimple_rnn_8_inputsimple_rnn_8_195110simple_rnn_8_195112simple_rnn_8_195114*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????x?*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_simple_rnn_8_layer_call_and_return_conditional_losses_1947162&
$simple_rnn_8/StatefulPartitionedCall?
activation_20/PartitionedCallPartitionedCall-simple_rnn_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????x?* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_activation_20_layer_call_and_return_conditional_losses_1947512
activation_20/PartitionedCall?
dropout_8/PartitionedCallPartitionedCall&activation_20/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????x?* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_8_layer_call_and_return_conditional_losses_1947732
dropout_8/PartitionedCall?
$simple_rnn_9/StatefulPartitionedCallStatefulPartitionedCall"dropout_8/PartitionedCall:output:0simple_rnn_9_195119simple_rnn_9_195121simple_rnn_9_195123*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_simple_rnn_9_layer_call_and_return_conditional_losses_1950092&
$simple_rnn_9/StatefulPartitionedCall?
activation_21/PartitionedCallPartitionedCall-simple_rnn_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_activation_21_layer_call_and_return_conditional_losses_1950442
activation_21/PartitionedCall?
dropout_9/PartitionedCallPartitionedCall&activation_21/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_9_layer_call_and_return_conditional_losses_1950662
dropout_9/PartitionedCall?
 dense_28/StatefulPartitionedCallStatefulPartitionedCall"dropout_9/PartitionedCall:output:0dense_28_195128dense_28_195130*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????<*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_28_layer_call_and_return_conditional_losses_1950902"
 dense_28/StatefulPartitionedCall?
IdentityIdentity)dense_28/StatefulPartitionedCall:output:0!^dense_28/StatefulPartitionedCall%^simple_rnn_8/StatefulPartitionedCall%^simple_rnn_9/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????<2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:?????????x::::::::2D
 dense_28/StatefulPartitionedCall dense_28/StatefulPartitionedCall2L
$simple_rnn_8/StatefulPartitionedCall$simple_rnn_8/StatefulPartitionedCall2L
$simple_rnn_9/StatefulPartitionedCall$simple_rnn_9/StatefulPartitionedCall:_ [
+
_output_shapes
:?????????x
,
_user_specified_namesimple_rnn_8_input
?C
?
simple_rnn_9_while_body_1956576
2simple_rnn_9_while_simple_rnn_9_while_loop_counter<
8simple_rnn_9_while_simple_rnn_9_while_maximum_iterations"
simple_rnn_9_while_placeholder$
 simple_rnn_9_while_placeholder_1$
 simple_rnn_9_while_placeholder_25
1simple_rnn_9_while_simple_rnn_9_strided_slice_1_0q
msimple_rnn_9_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_9_tensorarrayunstack_tensorlistfromtensor_0J
Fsimple_rnn_9_while_simple_rnn_cell_17_matmul_readvariableop_resource_0K
Gsimple_rnn_9_while_simple_rnn_cell_17_biasadd_readvariableop_resource_0L
Hsimple_rnn_9_while_simple_rnn_cell_17_matmul_1_readvariableop_resource_0
simple_rnn_9_while_identity!
simple_rnn_9_while_identity_1!
simple_rnn_9_while_identity_2!
simple_rnn_9_while_identity_3!
simple_rnn_9_while_identity_43
/simple_rnn_9_while_simple_rnn_9_strided_slice_1o
ksimple_rnn_9_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_9_tensorarrayunstack_tensorlistfromtensorH
Dsimple_rnn_9_while_simple_rnn_cell_17_matmul_readvariableop_resourceI
Esimple_rnn_9_while_simple_rnn_cell_17_biasadd_readvariableop_resourceJ
Fsimple_rnn_9_while_simple_rnn_cell_17_matmul_1_readvariableop_resource??<simple_rnn_9/while/simple_rnn_cell_17/BiasAdd/ReadVariableOp?;simple_rnn_9/while/simple_rnn_cell_17/MatMul/ReadVariableOp?=simple_rnn_9/while/simple_rnn_cell_17/MatMul_1/ReadVariableOp?
Dsimple_rnn_9/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2F
Dsimple_rnn_9/while/TensorArrayV2Read/TensorListGetItem/element_shape?
6simple_rnn_9/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemmsimple_rnn_9_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_9_tensorarrayunstack_tensorlistfromtensor_0simple_rnn_9_while_placeholderMsimple_rnn_9/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype028
6simple_rnn_9/while/TensorArrayV2Read/TensorListGetItem?
;simple_rnn_9/while/simple_rnn_cell_17/MatMul/ReadVariableOpReadVariableOpFsimple_rnn_9_while_simple_rnn_cell_17_matmul_readvariableop_resource_0*
_output_shapes
:	?2*
dtype02=
;simple_rnn_9/while/simple_rnn_cell_17/MatMul/ReadVariableOp?
,simple_rnn_9/while/simple_rnn_cell_17/MatMul	MLCMatMul=simple_rnn_9/while/TensorArrayV2Read/TensorListGetItem:item:0Csimple_rnn_9/while/simple_rnn_cell_17/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22.
,simple_rnn_9/while/simple_rnn_cell_17/MatMul?
<simple_rnn_9/while/simple_rnn_cell_17/BiasAdd/ReadVariableOpReadVariableOpGsimple_rnn_9_while_simple_rnn_cell_17_biasadd_readvariableop_resource_0*
_output_shapes
:2*
dtype02>
<simple_rnn_9/while/simple_rnn_cell_17/BiasAdd/ReadVariableOp?
-simple_rnn_9/while/simple_rnn_cell_17/BiasAddBiasAdd6simple_rnn_9/while/simple_rnn_cell_17/MatMul:product:0Dsimple_rnn_9/while/simple_rnn_cell_17/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22/
-simple_rnn_9/while/simple_rnn_cell_17/BiasAdd?
=simple_rnn_9/while/simple_rnn_cell_17/MatMul_1/ReadVariableOpReadVariableOpHsimple_rnn_9_while_simple_rnn_cell_17_matmul_1_readvariableop_resource_0*
_output_shapes

:22*
dtype02?
=simple_rnn_9/while/simple_rnn_cell_17/MatMul_1/ReadVariableOp?
.simple_rnn_9/while/simple_rnn_cell_17/MatMul_1	MLCMatMul simple_rnn_9_while_placeholder_2Esimple_rnn_9/while/simple_rnn_cell_17/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????220
.simple_rnn_9/while/simple_rnn_cell_17/MatMul_1?
)simple_rnn_9/while/simple_rnn_cell_17/addAddV26simple_rnn_9/while/simple_rnn_cell_17/BiasAdd:output:08simple_rnn_9/while/simple_rnn_cell_17/MatMul_1:product:0*
T0*'
_output_shapes
:?????????22+
)simple_rnn_9/while/simple_rnn_cell_17/add?
*simple_rnn_9/while/simple_rnn_cell_17/TanhTanh-simple_rnn_9/while/simple_rnn_cell_17/add:z:0*
T0*'
_output_shapes
:?????????22,
*simple_rnn_9/while/simple_rnn_cell_17/Tanh?
7simple_rnn_9/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem simple_rnn_9_while_placeholder_1simple_rnn_9_while_placeholder.simple_rnn_9/while/simple_rnn_cell_17/Tanh:y:0*
_output_shapes
: *
element_dtype029
7simple_rnn_9/while/TensorArrayV2Write/TensorListSetItemv
simple_rnn_9/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
simple_rnn_9/while/add/y?
simple_rnn_9/while/addAddV2simple_rnn_9_while_placeholder!simple_rnn_9/while/add/y:output:0*
T0*
_output_shapes
: 2
simple_rnn_9/while/addz
simple_rnn_9/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
simple_rnn_9/while/add_1/y?
simple_rnn_9/while/add_1AddV22simple_rnn_9_while_simple_rnn_9_while_loop_counter#simple_rnn_9/while/add_1/y:output:0*
T0*
_output_shapes
: 2
simple_rnn_9/while/add_1?
simple_rnn_9/while/IdentityIdentitysimple_rnn_9/while/add_1:z:0=^simple_rnn_9/while/simple_rnn_cell_17/BiasAdd/ReadVariableOp<^simple_rnn_9/while/simple_rnn_cell_17/MatMul/ReadVariableOp>^simple_rnn_9/while/simple_rnn_cell_17/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
simple_rnn_9/while/Identity?
simple_rnn_9/while/Identity_1Identity8simple_rnn_9_while_simple_rnn_9_while_maximum_iterations=^simple_rnn_9/while/simple_rnn_cell_17/BiasAdd/ReadVariableOp<^simple_rnn_9/while/simple_rnn_cell_17/MatMul/ReadVariableOp>^simple_rnn_9/while/simple_rnn_cell_17/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
simple_rnn_9/while/Identity_1?
simple_rnn_9/while/Identity_2Identitysimple_rnn_9/while/add:z:0=^simple_rnn_9/while/simple_rnn_cell_17/BiasAdd/ReadVariableOp<^simple_rnn_9/while/simple_rnn_cell_17/MatMul/ReadVariableOp>^simple_rnn_9/while/simple_rnn_cell_17/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
simple_rnn_9/while/Identity_2?
simple_rnn_9/while/Identity_3IdentityGsimple_rnn_9/while/TensorArrayV2Write/TensorListSetItem:output_handle:0=^simple_rnn_9/while/simple_rnn_cell_17/BiasAdd/ReadVariableOp<^simple_rnn_9/while/simple_rnn_cell_17/MatMul/ReadVariableOp>^simple_rnn_9/while/simple_rnn_cell_17/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
simple_rnn_9/while/Identity_3?
simple_rnn_9/while/Identity_4Identity.simple_rnn_9/while/simple_rnn_cell_17/Tanh:y:0=^simple_rnn_9/while/simple_rnn_cell_17/BiasAdd/ReadVariableOp<^simple_rnn_9/while/simple_rnn_cell_17/MatMul/ReadVariableOp>^simple_rnn_9/while/simple_rnn_cell_17/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????22
simple_rnn_9/while/Identity_4"C
simple_rnn_9_while_identity$simple_rnn_9/while/Identity:output:0"G
simple_rnn_9_while_identity_1&simple_rnn_9/while/Identity_1:output:0"G
simple_rnn_9_while_identity_2&simple_rnn_9/while/Identity_2:output:0"G
simple_rnn_9_while_identity_3&simple_rnn_9/while/Identity_3:output:0"G
simple_rnn_9_while_identity_4&simple_rnn_9/while/Identity_4:output:0"d
/simple_rnn_9_while_simple_rnn_9_strided_slice_11simple_rnn_9_while_simple_rnn_9_strided_slice_1_0"?
Esimple_rnn_9_while_simple_rnn_cell_17_biasadd_readvariableop_resourceGsimple_rnn_9_while_simple_rnn_cell_17_biasadd_readvariableop_resource_0"?
Fsimple_rnn_9_while_simple_rnn_cell_17_matmul_1_readvariableop_resourceHsimple_rnn_9_while_simple_rnn_cell_17_matmul_1_readvariableop_resource_0"?
Dsimple_rnn_9_while_simple_rnn_cell_17_matmul_readvariableop_resourceFsimple_rnn_9_while_simple_rnn_cell_17_matmul_readvariableop_resource_0"?
ksimple_rnn_9_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_9_tensorarrayunstack_tensorlistfromtensormsimple_rnn_9_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_9_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :?????????2: : :::2|
<simple_rnn_9/while/simple_rnn_cell_17/BiasAdd/ReadVariableOp<simple_rnn_9/while/simple_rnn_cell_17/BiasAdd/ReadVariableOp2z
;simple_rnn_9/while/simple_rnn_cell_17/MatMul/ReadVariableOp;simple_rnn_9/while/simple_rnn_cell_17/MatMul/ReadVariableOp2~
=simple_rnn_9/while/simple_rnn_cell_17/MatMul_1/ReadVariableOp=simple_rnn_9/while/simple_rnn_cell_17/MatMul_1/ReadVariableOp: 
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
:?????????2:

_output_shapes
: :

_output_shapes
: 
?4
?
while_body_195932
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0=
9while_simple_rnn_cell_16_matmul_readvariableop_resource_0>
:while_simple_rnn_cell_16_biasadd_readvariableop_resource_0?
;while_simple_rnn_cell_16_matmul_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor;
7while_simple_rnn_cell_16_matmul_readvariableop_resource<
8while_simple_rnn_cell_16_biasadd_readvariableop_resource=
9while_simple_rnn_cell_16_matmul_1_readvariableop_resource??/while/simple_rnn_cell_16/BiasAdd/ReadVariableOp?.while/simple_rnn_cell_16/MatMul/ReadVariableOp?0while/simple_rnn_cell_16/MatMul_1/ReadVariableOp?
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
.while/simple_rnn_cell_16/MatMul/ReadVariableOpReadVariableOp9while_simple_rnn_cell_16_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype020
.while/simple_rnn_cell_16/MatMul/ReadVariableOp?
while/simple_rnn_cell_16/MatMul	MLCMatMul0while/TensorArrayV2Read/TensorListGetItem:item:06while/simple_rnn_cell_16/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2!
while/simple_rnn_cell_16/MatMul?
/while/simple_rnn_cell_16/BiasAdd/ReadVariableOpReadVariableOp:while_simple_rnn_cell_16_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype021
/while/simple_rnn_cell_16/BiasAdd/ReadVariableOp?
 while/simple_rnn_cell_16/BiasAddBiasAdd)while/simple_rnn_cell_16/MatMul:product:07while/simple_rnn_cell_16/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2"
 while/simple_rnn_cell_16/BiasAdd?
0while/simple_rnn_cell_16/MatMul_1/ReadVariableOpReadVariableOp;while_simple_rnn_cell_16_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype022
0while/simple_rnn_cell_16/MatMul_1/ReadVariableOp?
!while/simple_rnn_cell_16/MatMul_1	MLCMatMulwhile_placeholder_28while/simple_rnn_cell_16/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2#
!while/simple_rnn_cell_16/MatMul_1?
while/simple_rnn_cell_16/addAddV2)while/simple_rnn_cell_16/BiasAdd:output:0+while/simple_rnn_cell_16/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
while/simple_rnn_cell_16/add?
while/simple_rnn_cell_16/TanhTanh while/simple_rnn_cell_16/add:z:0*
T0*(
_output_shapes
:??????????2
while/simple_rnn_cell_16/Tanh?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder!while/simple_rnn_cell_16/Tanh:y:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1?
while/IdentityIdentitywhile/add_1:z:00^while/simple_rnn_cell_16/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_16/MatMul/ReadVariableOp1^while/simple_rnn_cell_16/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations0^while/simple_rnn_cell_16/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_16/MatMul/ReadVariableOp1^while/simple_rnn_cell_16/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:00^while/simple_rnn_cell_16/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_16/MatMul/ReadVariableOp1^while/simple_rnn_cell_16/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:00^while/simple_rnn_cell_16/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_16/MatMul/ReadVariableOp1^while/simple_rnn_cell_16/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity!while/simple_rnn_cell_16/Tanh:y:00^while/simple_rnn_cell_16/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_16/MatMul/ReadVariableOp1^while/simple_rnn_cell_16/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2
while/Identity_4")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"v
8while_simple_rnn_cell_16_biasadd_readvariableop_resource:while_simple_rnn_cell_16_biasadd_readvariableop_resource_0"x
9while_simple_rnn_cell_16_matmul_1_readvariableop_resource;while_simple_rnn_cell_16_matmul_1_readvariableop_resource_0"t
7while_simple_rnn_cell_16_matmul_readvariableop_resource9while_simple_rnn_cell_16_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*?
_input_shapes.
,: : : : :??????????: : :::2b
/while/simple_rnn_cell_16/BiasAdd/ReadVariableOp/while/simple_rnn_cell_16/BiasAdd/ReadVariableOp2`
.while/simple_rnn_cell_16/MatMul/ReadVariableOp.while/simple_rnn_cell_16/MatMul/ReadVariableOp2d
0while/simple_rnn_cell_16/MatMul_1/ReadVariableOp0while/simple_rnn_cell_16/MatMul_1/ReadVariableOp: 
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
: :.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?
?
while_cond_196345
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_196345___redundant_placeholder04
0while_while_cond_196345___redundant_placeholder14
0while_while_cond_196345___redundant_placeholder24
0while_while_cond_196345___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
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
while_identitywhile/Identity:output:0*@
_input_shapes/
-: : : : :?????????2: ::::: 
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
:?????????2:

_output_shapes
: :

_output_shapes
:
?
?
-__inference_simple_rnn_8_layer_call_fn_196255

inputs
unknown
	unknown_0
	unknown_1
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????x?*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_simple_rnn_8_layer_call_and_return_conditional_losses_1946042
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:?????????x?2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????x:::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????x
 
_user_specified_nameinputs
?
?
I__inference_sequential_24_layer_call_and_return_conditional_losses_195212

inputs
simple_rnn_8_195188
simple_rnn_8_195190
simple_rnn_8_195192
simple_rnn_9_195197
simple_rnn_9_195199
simple_rnn_9_195201
dense_28_195206
dense_28_195208
identity?? dense_28/StatefulPartitionedCall?$simple_rnn_8/StatefulPartitionedCall?$simple_rnn_9/StatefulPartitionedCall?
$simple_rnn_8/StatefulPartitionedCallStatefulPartitionedCallinputssimple_rnn_8_195188simple_rnn_8_195190simple_rnn_8_195192*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????x?*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_simple_rnn_8_layer_call_and_return_conditional_losses_1947162&
$simple_rnn_8/StatefulPartitionedCall?
activation_20/PartitionedCallPartitionedCall-simple_rnn_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????x?* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_activation_20_layer_call_and_return_conditional_losses_1947512
activation_20/PartitionedCall?
dropout_8/PartitionedCallPartitionedCall&activation_20/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????x?* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_8_layer_call_and_return_conditional_losses_1947732
dropout_8/PartitionedCall?
$simple_rnn_9/StatefulPartitionedCallStatefulPartitionedCall"dropout_8/PartitionedCall:output:0simple_rnn_9_195197simple_rnn_9_195199simple_rnn_9_195201*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_simple_rnn_9_layer_call_and_return_conditional_losses_1950092&
$simple_rnn_9/StatefulPartitionedCall?
activation_21/PartitionedCallPartitionedCall-simple_rnn_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_activation_21_layer_call_and_return_conditional_losses_1950442
activation_21/PartitionedCall?
dropout_9/PartitionedCallPartitionedCall&activation_21/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_9_layer_call_and_return_conditional_losses_1950662
dropout_9/PartitionedCall?
 dense_28/StatefulPartitionedCallStatefulPartitionedCall"dropout_9/PartitionedCall:output:0dense_28_195206dense_28_195208*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????<*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_28_layer_call_and_return_conditional_losses_1950902"
 dense_28/StatefulPartitionedCall?
IdentityIdentity)dense_28/StatefulPartitionedCall:output:0!^dense_28/StatefulPartitionedCall%^simple_rnn_8/StatefulPartitionedCall%^simple_rnn_9/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????<2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:?????????x::::::::2D
 dense_28/StatefulPartitionedCall dense_28/StatefulPartitionedCall2L
$simple_rnn_8/StatefulPartitionedCall$simple_rnn_8/StatefulPartitionedCall2L
$simple_rnn_9/StatefulPartitionedCall$simple_rnn_9/StatefulPartitionedCall:S O
+
_output_shapes
:?????????x
 
_user_specified_nameinputs
?
?
.__inference_sequential_24_layer_call_fn_195231
simple_rnn_8_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallsimple_rnn_8_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????<**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_sequential_24_layer_call_and_return_conditional_losses_1952122
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????<2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:?????????x::::::::22
StatefulPartitionedCallStatefulPartitionedCall:_ [
+
_output_shapes
:?????????x
,
_user_specified_namesimple_rnn_8_input
?
c
E__inference_dropout_8_layer_call_and_return_conditional_losses_196290

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:?????????x?2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:?????????x?2

Identity_1"!

identity_1Identity_1:output:0*+
_input_shapes
:?????????x?:T P
,
_output_shapes
:?????????x?
 
_user_specified_nameinputs
?
e
I__inference_activation_21_layer_call_and_return_conditional_losses_196797

inputs
identityN
ReluReluinputs*
T0*'
_output_shapes
:?????????22
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????2:O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs
?

?
simple_rnn_9_while_cond_1956566
2simple_rnn_9_while_simple_rnn_9_while_loop_counter<
8simple_rnn_9_while_simple_rnn_9_while_maximum_iterations"
simple_rnn_9_while_placeholder$
 simple_rnn_9_while_placeholder_1$
 simple_rnn_9_while_placeholder_28
4simple_rnn_9_while_less_simple_rnn_9_strided_slice_1N
Jsimple_rnn_9_while_simple_rnn_9_while_cond_195656___redundant_placeholder0N
Jsimple_rnn_9_while_simple_rnn_9_while_cond_195656___redundant_placeholder1N
Jsimple_rnn_9_while_simple_rnn_9_while_cond_195656___redundant_placeholder2N
Jsimple_rnn_9_while_simple_rnn_9_while_cond_195656___redundant_placeholder3
simple_rnn_9_while_identity
?
simple_rnn_9/while/LessLesssimple_rnn_9_while_placeholder4simple_rnn_9_while_less_simple_rnn_9_strided_slice_1*
T0*
_output_shapes
: 2
simple_rnn_9/while/Less?
simple_rnn_9/while/IdentityIdentitysimple_rnn_9/while/Less:z:0*
T0
*
_output_shapes
: 2
simple_rnn_9/while/Identity"C
simple_rnn_9_while_identity$simple_rnn_9/while/Identity:output:0*@
_input_shapes/
-: : : : :?????????2: ::::: 
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
:?????????2:

_output_shapes
: :

_output_shapes
:
?
a
E__inference_dropout_9_layer_call_and_return_conditional_losses_196811

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:?????????22
dropout/Mula
dropout/rateConst*
_output_shapes
: *
dtype0*
valueB
 *???=2
dropout/rate^
dropout/seedConst*
_output_shapes
: *
dtype0*
value	B : 2
dropout/seed?
dropout
MLCDropoutinputsdropout/rate:output:0dropout/seed:output:0*
T0*'
_output_shapes
:?????????22	
dropoutd
IdentityIdentitydropout:output:0*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????2:O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs
?#
?
while_body_193788
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0%
!while_simple_rnn_cell_16_193810_0%
!while_simple_rnn_cell_16_193812_0%
!while_simple_rnn_cell_16_193814_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor#
while_simple_rnn_cell_16_193810#
while_simple_rnn_cell_16_193812#
while_simple_rnn_cell_16_193814??0while/simple_rnn_cell_16/StatefulPartitionedCall?
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
)while/TensorArrayV2Read/TensorListGetItem?
0while/simple_rnn_cell_16/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2!while_simple_rnn_cell_16_193810_0!while_simple_rnn_cell_16_193812_0!while_simple_rnn_cell_16_193814_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:??????????:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_simple_rnn_cell_16_layer_call_and_return_conditional_losses_19351422
0while/simple_rnn_cell_16/StatefulPartitionedCall?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder9while/simple_rnn_cell_16/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1?
while/IdentityIdentitywhile/add_1:z:01^while/simple_rnn_cell_16/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations1^while/simple_rnn_cell_16/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:01^while/simple_rnn_cell_16/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:01^while/simple_rnn_cell_16/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity9while/simple_rnn_cell_16/StatefulPartitionedCall:output:11^while/simple_rnn_cell_16/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2
while/Identity_4")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"D
while_simple_rnn_cell_16_193810!while_simple_rnn_cell_16_193810_0"D
while_simple_rnn_cell_16_193812!while_simple_rnn_cell_16_193812_0"D
while_simple_rnn_cell_16_193814!while_simple_rnn_cell_16_193814_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*?
_input_shapes.
,: : : : :??????????: : :::2d
0while/simple_rnn_cell_16/StatefulPartitionedCall0while/simple_rnn_cell_16/StatefulPartitionedCall: 
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
: :.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?#
?
while_body_193905
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0%
!while_simple_rnn_cell_16_193927_0%
!while_simple_rnn_cell_16_193929_0%
!while_simple_rnn_cell_16_193931_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor#
while_simple_rnn_cell_16_193927#
while_simple_rnn_cell_16_193929#
while_simple_rnn_cell_16_193931??0while/simple_rnn_cell_16/StatefulPartitionedCall?
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
)while/TensorArrayV2Read/TensorListGetItem?
0while/simple_rnn_cell_16/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2!while_simple_rnn_cell_16_193927_0!while_simple_rnn_cell_16_193929_0!while_simple_rnn_cell_16_193931_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:??????????:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_simple_rnn_cell_16_layer_call_and_return_conditional_losses_19353122
0while/simple_rnn_cell_16/StatefulPartitionedCall?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder9while/simple_rnn_cell_16/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1?
while/IdentityIdentitywhile/add_1:z:01^while/simple_rnn_cell_16/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations1^while/simple_rnn_cell_16/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:01^while/simple_rnn_cell_16/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:01^while/simple_rnn_cell_16/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity9while/simple_rnn_cell_16/StatefulPartitionedCall:output:11^while/simple_rnn_cell_16/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2
while/Identity_4")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"D
while_simple_rnn_cell_16_193927!while_simple_rnn_cell_16_193927_0"D
while_simple_rnn_cell_16_193929!while_simple_rnn_cell_16_193929_0"D
while_simple_rnn_cell_16_193931!while_simple_rnn_cell_16_193931_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*?
_input_shapes.
,: : : : :??????????: : :::2d
0while/simple_rnn_cell_16/StatefulPartitionedCall0while/simple_rnn_cell_16/StatefulPartitionedCall: 
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
: :.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?

?
simple_rnn_9_while_cond_1954216
2simple_rnn_9_while_simple_rnn_9_while_loop_counter<
8simple_rnn_9_while_simple_rnn_9_while_maximum_iterations"
simple_rnn_9_while_placeholder$
 simple_rnn_9_while_placeholder_1$
 simple_rnn_9_while_placeholder_28
4simple_rnn_9_while_less_simple_rnn_9_strided_slice_1N
Jsimple_rnn_9_while_simple_rnn_9_while_cond_195421___redundant_placeholder0N
Jsimple_rnn_9_while_simple_rnn_9_while_cond_195421___redundant_placeholder1N
Jsimple_rnn_9_while_simple_rnn_9_while_cond_195421___redundant_placeholder2N
Jsimple_rnn_9_while_simple_rnn_9_while_cond_195421___redundant_placeholder3
simple_rnn_9_while_identity
?
simple_rnn_9/while/LessLesssimple_rnn_9_while_placeholder4simple_rnn_9_while_less_simple_rnn_9_strided_slice_1*
T0*
_output_shapes
: 2
simple_rnn_9/while/Less?
simple_rnn_9/while/IdentityIdentitysimple_rnn_9/while/Less:z:0*
T0
*
_output_shapes
: 2
simple_rnn_9/while/Identity"C
simple_rnn_9_while_identity$simple_rnn_9/while/Identity:output:0*@
_input_shapes/
-: : : : :?????????2: ::::: 
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
:?????????2:

_output_shapes
: :

_output_shapes
:
?
J
.__inference_activation_20_layer_call_fn_196276

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????x?* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_activation_20_layer_call_and_return_conditional_losses_1947512
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:?????????x?2

Identity"
identityIdentity:output:0*+
_input_shapes
:?????????x?:T P
,
_output_shapes
:?????????x?
 
_user_specified_nameinputs
?
?
.__inference_sequential_24_layer_call_fn_195774

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????<**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_sequential_24_layer_call_and_return_conditional_losses_1952122
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????<2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:?????????x::::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????x
 
_user_specified_nameinputs
?
?
-__inference_simple_rnn_8_layer_call_fn_196009
inputs_0
unknown
	unknown_0
	unknown_1
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:???????????????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_simple_rnn_8_layer_call_and_return_conditional_losses_1938512
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*5
_output_shapes#
!:???????????????????2

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
?
J
.__inference_activation_21_layer_call_fn_196802

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_activation_21_layer_call_and_return_conditional_losses_1950442
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????2:O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
U
simple_rnn_8_input?
$serving_default_simple_rnn_8_input:0?????????x<
dense_280
StatefulPartitionedCall:0?????????<tensorflow/serving/predict:??
?6
layer_with_weights-0
layer-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer-5
layer_with_weights-2
layer-6
	optimizer
	regularization_losses

	variables
trainable_variables
	keras_api

signatures
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses"?4
_tf_keras_sequential?3{"class_name": "Sequential", "name": "sequential_24", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_24", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 120, 5]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "simple_rnn_8_input"}}, {"class_name": "SimpleRNN", "config": {"name": "simple_rnn_8", "trainable": true, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 150, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0}}, {"class_name": "Activation", "config": {"name": "activation_20", "trainable": true, "dtype": "float32", "activation": "relu"}}, {"class_name": "Dropout", "config": {"name": "dropout_8", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}, {"class_name": "SimpleRNN", "config": {"name": "simple_rnn_9", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 50, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0}}, {"class_name": "Activation", "config": {"name": "activation_21", "trainable": true, "dtype": "float32", "activation": "relu"}}, {"class_name": "Dropout", "config": {"name": "dropout_9", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_28", "trainable": true, "dtype": "float32", "units": 60, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 5]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 120, 5]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_24", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 120, 5]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "simple_rnn_8_input"}}, {"class_name": "SimpleRNN", "config": {"name": "simple_rnn_8", "trainable": true, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 150, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0}}, {"class_name": "Activation", "config": {"name": "activation_20", "trainable": true, "dtype": "float32", "activation": "relu"}}, {"class_name": "Dropout", "config": {"name": "dropout_8", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}, {"class_name": "SimpleRNN", "config": {"name": "simple_rnn_9", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 50, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0}}, {"class_name": "Activation", "config": {"name": "activation_21", "trainable": true, "dtype": "float32", "activation": "relu"}}, {"class_name": "Dropout", "config": {"name": "dropout_9", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_28", "trainable": true, "dtype": "float32", "units": 60, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": "mean_squared_error", "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "mae", "dtype": "float32", "fn": "mean_absolute_error"}}, {"class_name": "MeanMetricWrapper", "config": {"name": "acc", "dtype": "float32", "fn": "categorical_accuracy"}}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?

cell

state_spec
regularization_losses
	variables
trainable_variables
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_rnn_layer?	{"class_name": "SimpleRNN", "name": "simple_rnn_8", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "simple_rnn_8", "trainable": true, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 150, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 5]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 120, 5]}}
?
regularization_losses
	variables
trainable_variables
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Activation", "name": "activation_20", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_20", "trainable": true, "dtype": "float32", "activation": "relu"}}
?
regularization_losses
	variables
trainable_variables
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout_8", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_8", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}
?
cell

state_spec
regularization_losses
	variables
 trainable_variables
!	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_rnn_layer?	{"class_name": "SimpleRNN", "name": "simple_rnn_9", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "simple_rnn_9", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 50, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 150]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 120, 150]}}
?
"regularization_losses
#	variables
$trainable_variables
%	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Activation", "name": "activation_21", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_21", "trainable": true, "dtype": "float32", "activation": "relu"}}
?
&regularization_losses
'	variables
(trainable_variables
)	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout_9", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_9", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}
?

*kernel
+bias
,regularization_losses
-	variables
.trainable_variables
/	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_28", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_28", "trainable": true, "dtype": "float32", "units": 60, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 50}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50]}}
?
0iter

1beta_1

2beta_2
	3decay
4learning_rate*m?+m?5m?6m?7m?8m?9m?:m?*v?+v?5v?6v?7v?8v?9v?:v?"
	optimizer
 "
trackable_list_wrapper
X
50
61
72
83
94
:5
*6
+7"
trackable_list_wrapper
X
50
61
72
83
94
:5
*6
+7"
trackable_list_wrapper
?
	regularization_losses
;layer_regularization_losses
<layer_metrics

	variables
=non_trainable_variables
trainable_variables
>metrics

?layers
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
?

5kernel
6recurrent_kernel
7bias
@regularization_losses
A	variables
Btrainable_variables
C	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "SimpleRNNCell", "name": "simple_rnn_cell_16", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "simple_rnn_cell_16", "trainable": true, "dtype": "float32", "units": 150, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
50
61
72"
trackable_list_wrapper
5
50
61
72"
trackable_list_wrapper
?
regularization_losses
Dlayer_regularization_losses
Elayer_metrics
	variables
Fnon_trainable_variables
trainable_variables
Gmetrics

Hlayers

Istates
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
regularization_losses
Jlayer_regularization_losses
Klayer_metrics
	variables
Lnon_trainable_variables
trainable_variables
Mmetrics

Nlayers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
regularization_losses
Olayer_regularization_losses
Player_metrics
	variables
Qnon_trainable_variables
trainable_variables
Rmetrics

Slayers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?

8kernel
9recurrent_kernel
:bias
Tregularization_losses
U	variables
Vtrainable_variables
W	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "SimpleRNNCell", "name": "simple_rnn_cell_17", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "simple_rnn_cell_17", "trainable": true, "dtype": "float32", "units": 50, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
80
91
:2"
trackable_list_wrapper
5
80
91
:2"
trackable_list_wrapper
?
regularization_losses
Xlayer_regularization_losses
Ylayer_metrics
	variables
Znon_trainable_variables
 trainable_variables
[metrics

\layers

]states
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
"regularization_losses
^layer_regularization_losses
_layer_metrics
#	variables
`non_trainable_variables
$trainable_variables
ametrics

blayers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
&regularization_losses
clayer_regularization_losses
dlayer_metrics
'	variables
enon_trainable_variables
(trainable_variables
fmetrics

glayers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
!:2<2dense_28/kernel
:<2dense_28/bias
 "
trackable_list_wrapper
.
*0
+1"
trackable_list_wrapper
.
*0
+1"
trackable_list_wrapper
?
,regularization_losses
hlayer_regularization_losses
ilayer_metrics
-	variables
jnon_trainable_variables
.trainable_variables
kmetrics

llayers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
9:7	?2&simple_rnn_8/simple_rnn_cell_16/kernel
D:B
??20simple_rnn_8/simple_rnn_cell_16/recurrent_kernel
3:1?2$simple_rnn_8/simple_rnn_cell_16/bias
9:7	?22&simple_rnn_9/simple_rnn_cell_17/kernel
B:@2220simple_rnn_9/simple_rnn_cell_17/recurrent_kernel
2:022$simple_rnn_9/simple_rnn_cell_17/bias
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
5
m0
n1
o2"
trackable_list_wrapper
Q
0
1
2
3
4
5
6"
trackable_list_wrapper
 "
trackable_list_wrapper
5
50
61
72"
trackable_list_wrapper
5
50
61
72"
trackable_list_wrapper
?
@regularization_losses
player_regularization_losses
qlayer_metrics
A	variables
rnon_trainable_variables
Btrainable_variables
smetrics

tlayers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
80
91
:2"
trackable_list_wrapper
5
80
91
:2"
trackable_list_wrapper
?
Tregularization_losses
ulayer_regularization_losses
vlayer_metrics
U	variables
wnon_trainable_variables
Vtrainable_variables
xmetrics

ylayers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
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
?
	ztotal
	{count
|	variables
}	keras_api"?
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
?
	~total
	count
?
_fn_kwargs
?	variables
?	keras_api"?
_tf_keras_metric?{"class_name": "MeanMetricWrapper", "name": "mae", "dtype": "float32", "config": {"name": "mae", "dtype": "float32", "fn": "mean_absolute_error"}}
?

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"?
_tf_keras_metric?{"class_name": "MeanMetricWrapper", "name": "acc", "dtype": "float32", "config": {"name": "acc", "dtype": "float32", "fn": "categorical_accuracy"}}
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
 "
trackable_list_wrapper
:  (2total
:  (2count
.
z0
{1"
trackable_list_wrapper
-
|	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
~0
1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
&:$2<2Adam/dense_28/kernel/m
 :<2Adam/dense_28/bias/m
>:<	?2-Adam/simple_rnn_8/simple_rnn_cell_16/kernel/m
I:G
??27Adam/simple_rnn_8/simple_rnn_cell_16/recurrent_kernel/m
8:6?2+Adam/simple_rnn_8/simple_rnn_cell_16/bias/m
>:<	?22-Adam/simple_rnn_9/simple_rnn_cell_17/kernel/m
G:E2227Adam/simple_rnn_9/simple_rnn_cell_17/recurrent_kernel/m
7:522+Adam/simple_rnn_9/simple_rnn_cell_17/bias/m
&:$2<2Adam/dense_28/kernel/v
 :<2Adam/dense_28/bias/v
>:<	?2-Adam/simple_rnn_8/simple_rnn_cell_16/kernel/v
I:G
??27Adam/simple_rnn_8/simple_rnn_cell_16/recurrent_kernel/v
8:6?2+Adam/simple_rnn_8/simple_rnn_cell_16/bias/v
>:<	?22-Adam/simple_rnn_9/simple_rnn_cell_17/kernel/v
G:E2227Adam/simple_rnn_9/simple_rnn_cell_17/recurrent_kernel/v
7:522+Adam/simple_rnn_9/simple_rnn_cell_17/bias/v
?2?
.__inference_sequential_24_layer_call_fn_195183
.__inference_sequential_24_layer_call_fn_195231
.__inference_sequential_24_layer_call_fn_195774
.__inference_sequential_24_layer_call_fn_195753?
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
!__inference__wrapped_model_193465?
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
annotations? *5?2
0?-
simple_rnn_8_input?????????x
?2?
I__inference_sequential_24_layer_call_and_return_conditional_losses_195732
I__inference_sequential_24_layer_call_and_return_conditional_losses_195501
I__inference_sequential_24_layer_call_and_return_conditional_losses_195107
I__inference_sequential_24_layer_call_and_return_conditional_losses_195134?
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
?2?
-__inference_simple_rnn_8_layer_call_fn_196266
-__inference_simple_rnn_8_layer_call_fn_196255
-__inference_simple_rnn_8_layer_call_fn_196009
-__inference_simple_rnn_8_layer_call_fn_196020?
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
?2?
H__inference_simple_rnn_8_layer_call_and_return_conditional_losses_195886
H__inference_simple_rnn_8_layer_call_and_return_conditional_losses_195998
H__inference_simple_rnn_8_layer_call_and_return_conditional_losses_196244
H__inference_simple_rnn_8_layer_call_and_return_conditional_losses_196132?
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
.__inference_activation_20_layer_call_fn_196276?
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
I__inference_activation_20_layer_call_and_return_conditional_losses_196271?
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
?2?
*__inference_dropout_8_layer_call_fn_196295
*__inference_dropout_8_layer_call_fn_196300?
???
FullArgSpec)
args!?
jself
jinputs

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
?2?
E__inference_dropout_8_layer_call_and_return_conditional_losses_196285
E__inference_dropout_8_layer_call_and_return_conditional_losses_196290?
???
FullArgSpec)
args!?
jself
jinputs

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
?2?
-__inference_simple_rnn_9_layer_call_fn_196546
-__inference_simple_rnn_9_layer_call_fn_196535
-__inference_simple_rnn_9_layer_call_fn_196781
-__inference_simple_rnn_9_layer_call_fn_196792?
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
?2?
H__inference_simple_rnn_9_layer_call_and_return_conditional_losses_196770
H__inference_simple_rnn_9_layer_call_and_return_conditional_losses_196412
H__inference_simple_rnn_9_layer_call_and_return_conditional_losses_196658
H__inference_simple_rnn_9_layer_call_and_return_conditional_losses_196524?
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
.__inference_activation_21_layer_call_fn_196802?
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
I__inference_activation_21_layer_call_and_return_conditional_losses_196797?
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
?2?
*__inference_dropout_9_layer_call_fn_196821
*__inference_dropout_9_layer_call_fn_196826?
???
FullArgSpec)
args!?
jself
jinputs

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
?2?
E__inference_dropout_9_layer_call_and_return_conditional_losses_196816
E__inference_dropout_9_layer_call_and_return_conditional_losses_196811?
???
FullArgSpec)
args!?
jself
jinputs

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
?2?
)__inference_dense_28_layer_call_fn_196846?
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
D__inference_dense_28_layer_call_and_return_conditional_losses_196837?
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
$__inference_signature_wrapper_195262simple_rnn_8_input"?
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
?2?
3__inference_simple_rnn_cell_16_layer_call_fn_196894
3__inference_simple_rnn_cell_16_layer_call_fn_196908?
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
?2?
N__inference_simple_rnn_cell_16_layer_call_and_return_conditional_losses_196863
N__inference_simple_rnn_cell_16_layer_call_and_return_conditional_losses_196880?
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
?2?
3__inference_simple_rnn_cell_17_layer_call_fn_196970
3__inference_simple_rnn_cell_17_layer_call_fn_196956?
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
?2?
N__inference_simple_rnn_cell_17_layer_call_and_return_conditional_losses_196942
N__inference_simple_rnn_cell_17_layer_call_and_return_conditional_losses_196925?
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
!__inference__wrapped_model_193465?5768:9*+??<
5?2
0?-
simple_rnn_8_input?????????x
? "3?0
.
dense_28"?
dense_28?????????<?
I__inference_activation_20_layer_call_and_return_conditional_losses_196271b4?1
*?'
%?"
inputs?????????x?
? "*?'
 ?
0?????????x?
? ?
.__inference_activation_20_layer_call_fn_196276U4?1
*?'
%?"
inputs?????????x?
? "??????????x??
I__inference_activation_21_layer_call_and_return_conditional_losses_196797X/?,
%?"
 ?
inputs?????????2
? "%?"
?
0?????????2
? }
.__inference_activation_21_layer_call_fn_196802K/?,
%?"
 ?
inputs?????????2
? "??????????2?
D__inference_dense_28_layer_call_and_return_conditional_losses_196837\*+/?,
%?"
 ?
inputs?????????2
? "%?"
?
0?????????<
? |
)__inference_dense_28_layer_call_fn_196846O*+/?,
%?"
 ?
inputs?????????2
? "??????????<?
E__inference_dropout_8_layer_call_and_return_conditional_losses_196285f8?5
.?+
%?"
inputs?????????x?
p
? "*?'
 ?
0?????????x?
? ?
E__inference_dropout_8_layer_call_and_return_conditional_losses_196290f8?5
.?+
%?"
inputs?????????x?
p 
? "*?'
 ?
0?????????x?
? ?
*__inference_dropout_8_layer_call_fn_196295Y8?5
.?+
%?"
inputs?????????x?
p
? "??????????x??
*__inference_dropout_8_layer_call_fn_196300Y8?5
.?+
%?"
inputs?????????x?
p 
? "??????????x??
E__inference_dropout_9_layer_call_and_return_conditional_losses_196811\3?0
)?&
 ?
inputs?????????2
p
? "%?"
?
0?????????2
? ?
E__inference_dropout_9_layer_call_and_return_conditional_losses_196816\3?0
)?&
 ?
inputs?????????2
p 
? "%?"
?
0?????????2
? }
*__inference_dropout_9_layer_call_fn_196821O3?0
)?&
 ?
inputs?????????2
p
? "??????????2}
*__inference_dropout_9_layer_call_fn_196826O3?0
)?&
 ?
inputs?????????2
p 
? "??????????2?
I__inference_sequential_24_layer_call_and_return_conditional_losses_195107z5768:9*+G?D
=?:
0?-
simple_rnn_8_input?????????x
p

 
? "%?"
?
0?????????<
? ?
I__inference_sequential_24_layer_call_and_return_conditional_losses_195134z5768:9*+G?D
=?:
0?-
simple_rnn_8_input?????????x
p 

 
? "%?"
?
0?????????<
? ?
I__inference_sequential_24_layer_call_and_return_conditional_losses_195501n5768:9*+;?8
1?.
$?!
inputs?????????x
p

 
? "%?"
?
0?????????<
? ?
I__inference_sequential_24_layer_call_and_return_conditional_losses_195732n5768:9*+;?8
1?.
$?!
inputs?????????x
p 

 
? "%?"
?
0?????????<
? ?
.__inference_sequential_24_layer_call_fn_195183m5768:9*+G?D
=?:
0?-
simple_rnn_8_input?????????x
p

 
? "??????????<?
.__inference_sequential_24_layer_call_fn_195231m5768:9*+G?D
=?:
0?-
simple_rnn_8_input?????????x
p 

 
? "??????????<?
.__inference_sequential_24_layer_call_fn_195753a5768:9*+;?8
1?.
$?!
inputs?????????x
p

 
? "??????????<?
.__inference_sequential_24_layer_call_fn_195774a5768:9*+;?8
1?.
$?!
inputs?????????x
p 

 
? "??????????<?
$__inference_signature_wrapper_195262?5768:9*+U?R
? 
K?H
F
simple_rnn_8_input0?-
simple_rnn_8_input?????????x"3?0
.
dense_28"?
dense_28?????????<?
H__inference_simple_rnn_8_layer_call_and_return_conditional_losses_195886?576O?L
E?B
4?1
/?,
inputs/0??????????????????

 
p

 
? "3?0
)?&
0???????????????????
? ?
H__inference_simple_rnn_8_layer_call_and_return_conditional_losses_195998?576O?L
E?B
4?1
/?,
inputs/0??????????????????

 
p 

 
? "3?0
)?&
0???????????????????
? ?
H__inference_simple_rnn_8_layer_call_and_return_conditional_losses_196132r576??<
5?2
$?!
inputs?????????x

 
p

 
? "*?'
 ?
0?????????x?
? ?
H__inference_simple_rnn_8_layer_call_and_return_conditional_losses_196244r576??<
5?2
$?!
inputs?????????x

 
p 

 
? "*?'
 ?
0?????????x?
? ?
-__inference_simple_rnn_8_layer_call_fn_196009~576O?L
E?B
4?1
/?,
inputs/0??????????????????

 
p

 
? "&?#????????????????????
-__inference_simple_rnn_8_layer_call_fn_196020~576O?L
E?B
4?1
/?,
inputs/0??????????????????

 
p 

 
? "&?#????????????????????
-__inference_simple_rnn_8_layer_call_fn_196255e576??<
5?2
$?!
inputs?????????x

 
p

 
? "??????????x??
-__inference_simple_rnn_8_layer_call_fn_196266e576??<
5?2
$?!
inputs?????????x

 
p 

 
? "??????????x??
H__inference_simple_rnn_9_layer_call_and_return_conditional_losses_196412n8:9@?=
6?3
%?"
inputs?????????x?

 
p

 
? "%?"
?
0?????????2
? ?
H__inference_simple_rnn_9_layer_call_and_return_conditional_losses_196524n8:9@?=
6?3
%?"
inputs?????????x?

 
p 

 
? "%?"
?
0?????????2
? ?
H__inference_simple_rnn_9_layer_call_and_return_conditional_losses_196658~8:9P?M
F?C
5?2
0?-
inputs/0???????????????????

 
p

 
? "%?"
?
0?????????2
? ?
H__inference_simple_rnn_9_layer_call_and_return_conditional_losses_196770~8:9P?M
F?C
5?2
0?-
inputs/0???????????????????

 
p 

 
? "%?"
?
0?????????2
? ?
-__inference_simple_rnn_9_layer_call_fn_196535a8:9@?=
6?3
%?"
inputs?????????x?

 
p

 
? "??????????2?
-__inference_simple_rnn_9_layer_call_fn_196546a8:9@?=
6?3
%?"
inputs?????????x?

 
p 

 
? "??????????2?
-__inference_simple_rnn_9_layer_call_fn_196781q8:9P?M
F?C
5?2
0?-
inputs/0???????????????????

 
p

 
? "??????????2?
-__inference_simple_rnn_9_layer_call_fn_196792q8:9P?M
F?C
5?2
0?-
inputs/0???????????????????

 
p 

 
? "??????????2?
N__inference_simple_rnn_cell_16_layer_call_and_return_conditional_losses_196863?576]?Z
S?P
 ?
inputs?????????
(?%
#? 
states/0??????????
p
? "T?Q
J?G
?
0/0??????????
%?"
 ?
0/1/0??????????
? ?
N__inference_simple_rnn_cell_16_layer_call_and_return_conditional_losses_196880?576]?Z
S?P
 ?
inputs?????????
(?%
#? 
states/0??????????
p 
? "T?Q
J?G
?
0/0??????????
%?"
 ?
0/1/0??????????
? ?
3__inference_simple_rnn_cell_16_layer_call_fn_196894?576]?Z
S?P
 ?
inputs?????????
(?%
#? 
states/0??????????
p
? "F?C
?
0??????????
#? 
?
1/0???????????
3__inference_simple_rnn_cell_16_layer_call_fn_196908?576]?Z
S?P
 ?
inputs?????????
(?%
#? 
states/0??????????
p 
? "F?C
?
0??????????
#? 
?
1/0???????????
N__inference_simple_rnn_cell_17_layer_call_and_return_conditional_losses_196925?8:9]?Z
S?P
!?
inputs??????????
'?$
"?
states/0?????????2
p
? "R?O
H?E
?
0/0?????????2
$?!
?
0/1/0?????????2
? ?
N__inference_simple_rnn_cell_17_layer_call_and_return_conditional_losses_196942?8:9]?Z
S?P
!?
inputs??????????
'?$
"?
states/0?????????2
p 
? "R?O
H?E
?
0/0?????????2
$?!
?
0/1/0?????????2
? ?
3__inference_simple_rnn_cell_17_layer_call_fn_196956?8:9]?Z
S?P
!?
inputs??????????
'?$
"?
states/0?????????2
p
? "D?A
?
0?????????2
"?
?
1/0?????????2?
3__inference_simple_rnn_cell_17_layer_call_fn_196970?8:9]?Z
S?P
!?
inputs??????????
'?$
"?
states/0?????????2
p 
? "D?A
?
0?????????2
"?
?
1/0?????????2