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
dense_14/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2$* 
shared_namedense_14/kernel
s
#dense_14/kernel/Read/ReadVariableOpReadVariableOpdense_14/kernel*
_output_shapes

:2$*
dtype0
r
dense_14/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:$*
shared_namedense_14/bias
k
!dense_14/bias/Read/ReadVariableOpReadVariableOpdense_14/bias*
_output_shapes
:$*
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
%simple_rnn_4/simple_rnn_cell_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*6
shared_name'%simple_rnn_4/simple_rnn_cell_8/kernel
?
9simple_rnn_4/simple_rnn_cell_8/kernel/Read/ReadVariableOpReadVariableOp%simple_rnn_4/simple_rnn_cell_8/kernel*
_output_shapes
:	?*
dtype0
?
/simple_rnn_4/simple_rnn_cell_8/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*@
shared_name1/simple_rnn_4/simple_rnn_cell_8/recurrent_kernel
?
Csimple_rnn_4/simple_rnn_cell_8/recurrent_kernel/Read/ReadVariableOpReadVariableOp/simple_rnn_4/simple_rnn_cell_8/recurrent_kernel* 
_output_shapes
:
??*
dtype0
?
#simple_rnn_4/simple_rnn_cell_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*4
shared_name%#simple_rnn_4/simple_rnn_cell_8/bias
?
7simple_rnn_4/simple_rnn_cell_8/bias/Read/ReadVariableOpReadVariableOp#simple_rnn_4/simple_rnn_cell_8/bias*
_output_shapes	
:?*
dtype0
?
%simple_rnn_5/simple_rnn_cell_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?2*6
shared_name'%simple_rnn_5/simple_rnn_cell_9/kernel
?
9simple_rnn_5/simple_rnn_cell_9/kernel/Read/ReadVariableOpReadVariableOp%simple_rnn_5/simple_rnn_cell_9/kernel*
_output_shapes
:	?2*
dtype0
?
/simple_rnn_5/simple_rnn_cell_9/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:22*@
shared_name1/simple_rnn_5/simple_rnn_cell_9/recurrent_kernel
?
Csimple_rnn_5/simple_rnn_cell_9/recurrent_kernel/Read/ReadVariableOpReadVariableOp/simple_rnn_5/simple_rnn_cell_9/recurrent_kernel*
_output_shapes

:22*
dtype0
?
#simple_rnn_5/simple_rnn_cell_9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*4
shared_name%#simple_rnn_5/simple_rnn_cell_9/bias
?
7simple_rnn_5/simple_rnn_cell_9/bias/Read/ReadVariableOpReadVariableOp#simple_rnn_5/simple_rnn_cell_9/bias*
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
Adam/dense_14/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2$*'
shared_nameAdam/dense_14/kernel/m
?
*Adam/dense_14/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_14/kernel/m*
_output_shapes

:2$*
dtype0
?
Adam/dense_14/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:$*%
shared_nameAdam/dense_14/bias/m
y
(Adam/dense_14/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_14/bias/m*
_output_shapes
:$*
dtype0
?
,Adam/simple_rnn_4/simple_rnn_cell_8/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*=
shared_name.,Adam/simple_rnn_4/simple_rnn_cell_8/kernel/m
?
@Adam/simple_rnn_4/simple_rnn_cell_8/kernel/m/Read/ReadVariableOpReadVariableOp,Adam/simple_rnn_4/simple_rnn_cell_8/kernel/m*
_output_shapes
:	?*
dtype0
?
6Adam/simple_rnn_4/simple_rnn_cell_8/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*G
shared_name86Adam/simple_rnn_4/simple_rnn_cell_8/recurrent_kernel/m
?
JAdam/simple_rnn_4/simple_rnn_cell_8/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp6Adam/simple_rnn_4/simple_rnn_cell_8/recurrent_kernel/m* 
_output_shapes
:
??*
dtype0
?
*Adam/simple_rnn_4/simple_rnn_cell_8/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*;
shared_name,*Adam/simple_rnn_4/simple_rnn_cell_8/bias/m
?
>Adam/simple_rnn_4/simple_rnn_cell_8/bias/m/Read/ReadVariableOpReadVariableOp*Adam/simple_rnn_4/simple_rnn_cell_8/bias/m*
_output_shapes	
:?*
dtype0
?
,Adam/simple_rnn_5/simple_rnn_cell_9/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?2*=
shared_name.,Adam/simple_rnn_5/simple_rnn_cell_9/kernel/m
?
@Adam/simple_rnn_5/simple_rnn_cell_9/kernel/m/Read/ReadVariableOpReadVariableOp,Adam/simple_rnn_5/simple_rnn_cell_9/kernel/m*
_output_shapes
:	?2*
dtype0
?
6Adam/simple_rnn_5/simple_rnn_cell_9/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:22*G
shared_name86Adam/simple_rnn_5/simple_rnn_cell_9/recurrent_kernel/m
?
JAdam/simple_rnn_5/simple_rnn_cell_9/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp6Adam/simple_rnn_5/simple_rnn_cell_9/recurrent_kernel/m*
_output_shapes

:22*
dtype0
?
*Adam/simple_rnn_5/simple_rnn_cell_9/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*;
shared_name,*Adam/simple_rnn_5/simple_rnn_cell_9/bias/m
?
>Adam/simple_rnn_5/simple_rnn_cell_9/bias/m/Read/ReadVariableOpReadVariableOp*Adam/simple_rnn_5/simple_rnn_cell_9/bias/m*
_output_shapes
:2*
dtype0
?
Adam/dense_14/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2$*'
shared_nameAdam/dense_14/kernel/v
?
*Adam/dense_14/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_14/kernel/v*
_output_shapes

:2$*
dtype0
?
Adam/dense_14/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:$*%
shared_nameAdam/dense_14/bias/v
y
(Adam/dense_14/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_14/bias/v*
_output_shapes
:$*
dtype0
?
,Adam/simple_rnn_4/simple_rnn_cell_8/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*=
shared_name.,Adam/simple_rnn_4/simple_rnn_cell_8/kernel/v
?
@Adam/simple_rnn_4/simple_rnn_cell_8/kernel/v/Read/ReadVariableOpReadVariableOp,Adam/simple_rnn_4/simple_rnn_cell_8/kernel/v*
_output_shapes
:	?*
dtype0
?
6Adam/simple_rnn_4/simple_rnn_cell_8/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*G
shared_name86Adam/simple_rnn_4/simple_rnn_cell_8/recurrent_kernel/v
?
JAdam/simple_rnn_4/simple_rnn_cell_8/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp6Adam/simple_rnn_4/simple_rnn_cell_8/recurrent_kernel/v* 
_output_shapes
:
??*
dtype0
?
*Adam/simple_rnn_4/simple_rnn_cell_8/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*;
shared_name,*Adam/simple_rnn_4/simple_rnn_cell_8/bias/v
?
>Adam/simple_rnn_4/simple_rnn_cell_8/bias/v/Read/ReadVariableOpReadVariableOp*Adam/simple_rnn_4/simple_rnn_cell_8/bias/v*
_output_shapes	
:?*
dtype0
?
,Adam/simple_rnn_5/simple_rnn_cell_9/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?2*=
shared_name.,Adam/simple_rnn_5/simple_rnn_cell_9/kernel/v
?
@Adam/simple_rnn_5/simple_rnn_cell_9/kernel/v/Read/ReadVariableOpReadVariableOp,Adam/simple_rnn_5/simple_rnn_cell_9/kernel/v*
_output_shapes
:	?2*
dtype0
?
6Adam/simple_rnn_5/simple_rnn_cell_9/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:22*G
shared_name86Adam/simple_rnn_5/simple_rnn_cell_9/recurrent_kernel/v
?
JAdam/simple_rnn_5/simple_rnn_cell_9/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp6Adam/simple_rnn_5/simple_rnn_cell_9/recurrent_kernel/v*
_output_shapes

:22*
dtype0
?
*Adam/simple_rnn_5/simple_rnn_cell_9/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*;
shared_name,*Adam/simple_rnn_5/simple_rnn_cell_9/bias/v
?
>Adam/simple_rnn_5/simple_rnn_cell_9/bias/v/Read/ReadVariableOpReadVariableOp*Adam/simple_rnn_5/simple_rnn_cell_9/bias/v*
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
VARIABLE_VALUEdense_14/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_14/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
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
a_
VARIABLE_VALUE%simple_rnn_4/simple_rnn_cell_8/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUE/simple_rnn_4/simple_rnn_cell_8/recurrent_kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUE#simple_rnn_4/simple_rnn_cell_8/bias&variables/2/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUE%simple_rnn_5/simple_rnn_cell_9/kernel&variables/3/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUE/simple_rnn_5/simple_rnn_cell_9/recurrent_kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUE#simple_rnn_5/simple_rnn_cell_9/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE
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
VARIABLE_VALUEAdam/dense_14/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_14/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE,Adam/simple_rnn_4/simple_rnn_cell_8/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE6Adam/simple_rnn_4/simple_rnn_cell_8/recurrent_kernel/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*Adam/simple_rnn_4/simple_rnn_cell_8/bias/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE,Adam/simple_rnn_5/simple_rnn_cell_9/kernel/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE6Adam/simple_rnn_5/simple_rnn_cell_9/recurrent_kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*Adam/simple_rnn_5/simple_rnn_cell_9/bias/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_14/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_14/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE,Adam/simple_rnn_4/simple_rnn_cell_8/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE6Adam/simple_rnn_4/simple_rnn_cell_8/recurrent_kernel/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*Adam/simple_rnn_4/simple_rnn_cell_8/bias/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE,Adam/simple_rnn_5/simple_rnn_cell_9/kernel/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE6Adam/simple_rnn_5/simple_rnn_cell_9/recurrent_kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*Adam/simple_rnn_5/simple_rnn_cell_9/bias/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
"serving_default_simple_rnn_4_inputPlaceholder*+
_output_shapes
:?????????H*
dtype0* 
shape:?????????H
?
StatefulPartitionedCallStatefulPartitionedCall"serving_default_simple_rnn_4_input%simple_rnn_4/simple_rnn_cell_8/kernel#simple_rnn_4/simple_rnn_cell_8/bias/simple_rnn_4/simple_rnn_cell_8/recurrent_kernel%simple_rnn_5/simple_rnn_cell_9/kernel#simple_rnn_5/simple_rnn_cell_9/bias/simple_rnn_5/simple_rnn_cell_9/recurrent_kerneldense_14/kerneldense_14/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????$**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *-
f(R&
$__inference_signature_wrapper_100620
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_14/kernel/Read/ReadVariableOp!dense_14/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp9simple_rnn_4/simple_rnn_cell_8/kernel/Read/ReadVariableOpCsimple_rnn_4/simple_rnn_cell_8/recurrent_kernel/Read/ReadVariableOp7simple_rnn_4/simple_rnn_cell_8/bias/Read/ReadVariableOp9simple_rnn_5/simple_rnn_cell_9/kernel/Read/ReadVariableOpCsimple_rnn_5/simple_rnn_cell_9/recurrent_kernel/Read/ReadVariableOp7simple_rnn_5/simple_rnn_cell_9/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal_2/Read/ReadVariableOpcount_2/Read/ReadVariableOp*Adam/dense_14/kernel/m/Read/ReadVariableOp(Adam/dense_14/bias/m/Read/ReadVariableOp@Adam/simple_rnn_4/simple_rnn_cell_8/kernel/m/Read/ReadVariableOpJAdam/simple_rnn_4/simple_rnn_cell_8/recurrent_kernel/m/Read/ReadVariableOp>Adam/simple_rnn_4/simple_rnn_cell_8/bias/m/Read/ReadVariableOp@Adam/simple_rnn_5/simple_rnn_cell_9/kernel/m/Read/ReadVariableOpJAdam/simple_rnn_5/simple_rnn_cell_9/recurrent_kernel/m/Read/ReadVariableOp>Adam/simple_rnn_5/simple_rnn_cell_9/bias/m/Read/ReadVariableOp*Adam/dense_14/kernel/v/Read/ReadVariableOp(Adam/dense_14/bias/v/Read/ReadVariableOp@Adam/simple_rnn_4/simple_rnn_cell_8/kernel/v/Read/ReadVariableOpJAdam/simple_rnn_4/simple_rnn_cell_8/recurrent_kernel/v/Read/ReadVariableOp>Adam/simple_rnn_4/simple_rnn_cell_8/bias/v/Read/ReadVariableOp@Adam/simple_rnn_5/simple_rnn_cell_9/kernel/v/Read/ReadVariableOpJAdam/simple_rnn_5/simple_rnn_cell_9/recurrent_kernel/v/Read/ReadVariableOp>Adam/simple_rnn_5/simple_rnn_cell_9/bias/v/Read/ReadVariableOpConst*0
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
__inference__traced_save_102456
?

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_14/kerneldense_14/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rate%simple_rnn_4/simple_rnn_cell_8/kernel/simple_rnn_4/simple_rnn_cell_8/recurrent_kernel#simple_rnn_4/simple_rnn_cell_8/bias%simple_rnn_5/simple_rnn_cell_9/kernel/simple_rnn_5/simple_rnn_cell_9/recurrent_kernel#simple_rnn_5/simple_rnn_cell_9/biastotalcounttotal_1count_1total_2count_2Adam/dense_14/kernel/mAdam/dense_14/bias/m,Adam/simple_rnn_4/simple_rnn_cell_8/kernel/m6Adam/simple_rnn_4/simple_rnn_cell_8/recurrent_kernel/m*Adam/simple_rnn_4/simple_rnn_cell_8/bias/m,Adam/simple_rnn_5/simple_rnn_cell_9/kernel/m6Adam/simple_rnn_5/simple_rnn_cell_9/recurrent_kernel/m*Adam/simple_rnn_5/simple_rnn_cell_9/bias/mAdam/dense_14/kernel/vAdam/dense_14/bias/v,Adam/simple_rnn_4/simple_rnn_cell_8/kernel/v6Adam/simple_rnn_4/simple_rnn_cell_8/recurrent_kernel/v*Adam/simple_rnn_4/simple_rnn_cell_8/bias/v,Adam/simple_rnn_5/simple_rnn_cell_9/kernel/v6Adam/simple_rnn_5/simple_rnn_cell_9/recurrent_kernel/v*Adam/simple_rnn_5/simple_rnn_cell_9/bias/v*/
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
"__inference__traced_restore_102571??
?
F
*__inference_dropout_5_layer_call_fn_102179

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
E__inference_dropout_5_layer_call_and_return_conditional_losses_1004192
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
?
?
while_cond_99895
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_99895___redundant_placeholder03
/while_while_cond_99895___redundant_placeholder13
/while_while_cond_99895___redundant_placeholder23
/while_while_cond_99895___redundant_placeholder3
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
L__inference_simple_rnn_cell_8_layer_call_and_return_conditional_losses_98872

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
L__inference_simple_rnn_cell_9_layer_call_and_return_conditional_losses_99401

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
?	
?
2__inference_simple_rnn_cell_8_layer_call_fn_102252

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
GPU 2J 8? *U
fPRN
L__inference_simple_rnn_cell_8_layer_call_and_return_conditional_losses_988722
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
?
while_cond_99657
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_99657___redundant_placeholder03
/while_while_cond_99657___redundant_placeholder13
/while_while_cond_99657___redundant_placeholder23
/while_while_cond_99657___redundant_placeholder3
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
?
?
M__inference_simple_rnn_cell_9_layer_call_and_return_conditional_losses_102300

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
?B
?
simple_rnn_4_while_body_1009056
2simple_rnn_4_while_simple_rnn_4_while_loop_counter<
8simple_rnn_4_while_simple_rnn_4_while_maximum_iterations"
simple_rnn_4_while_placeholder$
 simple_rnn_4_while_placeholder_1$
 simple_rnn_4_while_placeholder_25
1simple_rnn_4_while_simple_rnn_4_strided_slice_1_0q
msimple_rnn_4_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_4_tensorarrayunstack_tensorlistfromtensor_0I
Esimple_rnn_4_while_simple_rnn_cell_8_matmul_readvariableop_resource_0J
Fsimple_rnn_4_while_simple_rnn_cell_8_biasadd_readvariableop_resource_0K
Gsimple_rnn_4_while_simple_rnn_cell_8_matmul_1_readvariableop_resource_0
simple_rnn_4_while_identity!
simple_rnn_4_while_identity_1!
simple_rnn_4_while_identity_2!
simple_rnn_4_while_identity_3!
simple_rnn_4_while_identity_43
/simple_rnn_4_while_simple_rnn_4_strided_slice_1o
ksimple_rnn_4_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_4_tensorarrayunstack_tensorlistfromtensorG
Csimple_rnn_4_while_simple_rnn_cell_8_matmul_readvariableop_resourceH
Dsimple_rnn_4_while_simple_rnn_cell_8_biasadd_readvariableop_resourceI
Esimple_rnn_4_while_simple_rnn_cell_8_matmul_1_readvariableop_resource??;simple_rnn_4/while/simple_rnn_cell_8/BiasAdd/ReadVariableOp?:simple_rnn_4/while/simple_rnn_cell_8/MatMul/ReadVariableOp?<simple_rnn_4/while/simple_rnn_cell_8/MatMul_1/ReadVariableOp?
Dsimple_rnn_4/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2F
Dsimple_rnn_4/while/TensorArrayV2Read/TensorListGetItem/element_shape?
6simple_rnn_4/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemmsimple_rnn_4_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_4_tensorarrayunstack_tensorlistfromtensor_0simple_rnn_4_while_placeholderMsimple_rnn_4/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype028
6simple_rnn_4/while/TensorArrayV2Read/TensorListGetItem?
:simple_rnn_4/while/simple_rnn_cell_8/MatMul/ReadVariableOpReadVariableOpEsimple_rnn_4_while_simple_rnn_cell_8_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype02<
:simple_rnn_4/while/simple_rnn_cell_8/MatMul/ReadVariableOp?
+simple_rnn_4/while/simple_rnn_cell_8/MatMul	MLCMatMul=simple_rnn_4/while/TensorArrayV2Read/TensorListGetItem:item:0Bsimple_rnn_4/while/simple_rnn_cell_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2-
+simple_rnn_4/while/simple_rnn_cell_8/MatMul?
;simple_rnn_4/while/simple_rnn_cell_8/BiasAdd/ReadVariableOpReadVariableOpFsimple_rnn_4_while_simple_rnn_cell_8_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02=
;simple_rnn_4/while/simple_rnn_cell_8/BiasAdd/ReadVariableOp?
,simple_rnn_4/while/simple_rnn_cell_8/BiasAddBiasAdd5simple_rnn_4/while/simple_rnn_cell_8/MatMul:product:0Csimple_rnn_4/while/simple_rnn_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2.
,simple_rnn_4/while/simple_rnn_cell_8/BiasAdd?
<simple_rnn_4/while/simple_rnn_cell_8/MatMul_1/ReadVariableOpReadVariableOpGsimple_rnn_4_while_simple_rnn_cell_8_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02>
<simple_rnn_4/while/simple_rnn_cell_8/MatMul_1/ReadVariableOp?
-simple_rnn_4/while/simple_rnn_cell_8/MatMul_1	MLCMatMul simple_rnn_4_while_placeholder_2Dsimple_rnn_4/while/simple_rnn_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2/
-simple_rnn_4/while/simple_rnn_cell_8/MatMul_1?
(simple_rnn_4/while/simple_rnn_cell_8/addAddV25simple_rnn_4/while/simple_rnn_cell_8/BiasAdd:output:07simple_rnn_4/while/simple_rnn_cell_8/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2*
(simple_rnn_4/while/simple_rnn_cell_8/add?
)simple_rnn_4/while/simple_rnn_cell_8/TanhTanh,simple_rnn_4/while/simple_rnn_cell_8/add:z:0*
T0*(
_output_shapes
:??????????2+
)simple_rnn_4/while/simple_rnn_cell_8/Tanh?
7simple_rnn_4/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem simple_rnn_4_while_placeholder_1simple_rnn_4_while_placeholder-simple_rnn_4/while/simple_rnn_cell_8/Tanh:y:0*
_output_shapes
: *
element_dtype029
7simple_rnn_4/while/TensorArrayV2Write/TensorListSetItemv
simple_rnn_4/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
simple_rnn_4/while/add/y?
simple_rnn_4/while/addAddV2simple_rnn_4_while_placeholder!simple_rnn_4/while/add/y:output:0*
T0*
_output_shapes
: 2
simple_rnn_4/while/addz
simple_rnn_4/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
simple_rnn_4/while/add_1/y?
simple_rnn_4/while/add_1AddV22simple_rnn_4_while_simple_rnn_4_while_loop_counter#simple_rnn_4/while/add_1/y:output:0*
T0*
_output_shapes
: 2
simple_rnn_4/while/add_1?
simple_rnn_4/while/IdentityIdentitysimple_rnn_4/while/add_1:z:0<^simple_rnn_4/while/simple_rnn_cell_8/BiasAdd/ReadVariableOp;^simple_rnn_4/while/simple_rnn_cell_8/MatMul/ReadVariableOp=^simple_rnn_4/while/simple_rnn_cell_8/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
simple_rnn_4/while/Identity?
simple_rnn_4/while/Identity_1Identity8simple_rnn_4_while_simple_rnn_4_while_maximum_iterations<^simple_rnn_4/while/simple_rnn_cell_8/BiasAdd/ReadVariableOp;^simple_rnn_4/while/simple_rnn_cell_8/MatMul/ReadVariableOp=^simple_rnn_4/while/simple_rnn_cell_8/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
simple_rnn_4/while/Identity_1?
simple_rnn_4/while/Identity_2Identitysimple_rnn_4/while/add:z:0<^simple_rnn_4/while/simple_rnn_cell_8/BiasAdd/ReadVariableOp;^simple_rnn_4/while/simple_rnn_cell_8/MatMul/ReadVariableOp=^simple_rnn_4/while/simple_rnn_cell_8/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
simple_rnn_4/while/Identity_2?
simple_rnn_4/while/Identity_3IdentityGsimple_rnn_4/while/TensorArrayV2Write/TensorListSetItem:output_handle:0<^simple_rnn_4/while/simple_rnn_cell_8/BiasAdd/ReadVariableOp;^simple_rnn_4/while/simple_rnn_cell_8/MatMul/ReadVariableOp=^simple_rnn_4/while/simple_rnn_cell_8/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
simple_rnn_4/while/Identity_3?
simple_rnn_4/while/Identity_4Identity-simple_rnn_4/while/simple_rnn_cell_8/Tanh:y:0<^simple_rnn_4/while/simple_rnn_cell_8/BiasAdd/ReadVariableOp;^simple_rnn_4/while/simple_rnn_cell_8/MatMul/ReadVariableOp=^simple_rnn_4/while/simple_rnn_cell_8/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2
simple_rnn_4/while/Identity_4"C
simple_rnn_4_while_identity$simple_rnn_4/while/Identity:output:0"G
simple_rnn_4_while_identity_1&simple_rnn_4/while/Identity_1:output:0"G
simple_rnn_4_while_identity_2&simple_rnn_4/while/Identity_2:output:0"G
simple_rnn_4_while_identity_3&simple_rnn_4/while/Identity_3:output:0"G
simple_rnn_4_while_identity_4&simple_rnn_4/while/Identity_4:output:0"d
/simple_rnn_4_while_simple_rnn_4_strided_slice_11simple_rnn_4_while_simple_rnn_4_strided_slice_1_0"?
Dsimple_rnn_4_while_simple_rnn_cell_8_biasadd_readvariableop_resourceFsimple_rnn_4_while_simple_rnn_cell_8_biasadd_readvariableop_resource_0"?
Esimple_rnn_4_while_simple_rnn_cell_8_matmul_1_readvariableop_resourceGsimple_rnn_4_while_simple_rnn_cell_8_matmul_1_readvariableop_resource_0"?
Csimple_rnn_4_while_simple_rnn_cell_8_matmul_readvariableop_resourceEsimple_rnn_4_while_simple_rnn_cell_8_matmul_readvariableop_resource_0"?
ksimple_rnn_4_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_4_tensorarrayunstack_tensorlistfromtensormsimple_rnn_4_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_4_tensorarrayunstack_tensorlistfromtensor_0*?
_input_shapes.
,: : : : :??????????: : :::2z
;simple_rnn_4/while/simple_rnn_cell_8/BiasAdd/ReadVariableOp;simple_rnn_4/while/simple_rnn_cell_8/BiasAdd/ReadVariableOp2x
:simple_rnn_4/while/simple_rnn_cell_8/MatMul/ReadVariableOp:simple_rnn_4/while/simple_rnn_cell_8/MatMul/ReadVariableOp2|
<simple_rnn_4/while/simple_rnn_cell_8/MatMul_1/ReadVariableOp<simple_rnn_4/while/simple_rnn_cell_8/MatMul_1/ReadVariableOp: 
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
while_cond_99145
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_99145___redundant_placeholder03
/while_while_cond_99145___redundant_placeholder13
/while_while_cond_99145___redundant_placeholder23
/while_while_cond_99145___redundant_placeholder3
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
L__inference_simple_rnn_cell_9_layer_call_and_return_conditional_losses_99384

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
?B
?
simple_rnn_5_while_body_1007806
2simple_rnn_5_while_simple_rnn_5_while_loop_counter<
8simple_rnn_5_while_simple_rnn_5_while_maximum_iterations"
simple_rnn_5_while_placeholder$
 simple_rnn_5_while_placeholder_1$
 simple_rnn_5_while_placeholder_25
1simple_rnn_5_while_simple_rnn_5_strided_slice_1_0q
msimple_rnn_5_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_5_tensorarrayunstack_tensorlistfromtensor_0I
Esimple_rnn_5_while_simple_rnn_cell_9_matmul_readvariableop_resource_0J
Fsimple_rnn_5_while_simple_rnn_cell_9_biasadd_readvariableop_resource_0K
Gsimple_rnn_5_while_simple_rnn_cell_9_matmul_1_readvariableop_resource_0
simple_rnn_5_while_identity!
simple_rnn_5_while_identity_1!
simple_rnn_5_while_identity_2!
simple_rnn_5_while_identity_3!
simple_rnn_5_while_identity_43
/simple_rnn_5_while_simple_rnn_5_strided_slice_1o
ksimple_rnn_5_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_5_tensorarrayunstack_tensorlistfromtensorG
Csimple_rnn_5_while_simple_rnn_cell_9_matmul_readvariableop_resourceH
Dsimple_rnn_5_while_simple_rnn_cell_9_biasadd_readvariableop_resourceI
Esimple_rnn_5_while_simple_rnn_cell_9_matmul_1_readvariableop_resource??;simple_rnn_5/while/simple_rnn_cell_9/BiasAdd/ReadVariableOp?:simple_rnn_5/while/simple_rnn_cell_9/MatMul/ReadVariableOp?<simple_rnn_5/while/simple_rnn_cell_9/MatMul_1/ReadVariableOp?
Dsimple_rnn_5/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2F
Dsimple_rnn_5/while/TensorArrayV2Read/TensorListGetItem/element_shape?
6simple_rnn_5/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemmsimple_rnn_5_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_5_tensorarrayunstack_tensorlistfromtensor_0simple_rnn_5_while_placeholderMsimple_rnn_5/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype028
6simple_rnn_5/while/TensorArrayV2Read/TensorListGetItem?
:simple_rnn_5/while/simple_rnn_cell_9/MatMul/ReadVariableOpReadVariableOpEsimple_rnn_5_while_simple_rnn_cell_9_matmul_readvariableop_resource_0*
_output_shapes
:	?2*
dtype02<
:simple_rnn_5/while/simple_rnn_cell_9/MatMul/ReadVariableOp?
+simple_rnn_5/while/simple_rnn_cell_9/MatMul	MLCMatMul=simple_rnn_5/while/TensorArrayV2Read/TensorListGetItem:item:0Bsimple_rnn_5/while/simple_rnn_cell_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22-
+simple_rnn_5/while/simple_rnn_cell_9/MatMul?
;simple_rnn_5/while/simple_rnn_cell_9/BiasAdd/ReadVariableOpReadVariableOpFsimple_rnn_5_while_simple_rnn_cell_9_biasadd_readvariableop_resource_0*
_output_shapes
:2*
dtype02=
;simple_rnn_5/while/simple_rnn_cell_9/BiasAdd/ReadVariableOp?
,simple_rnn_5/while/simple_rnn_cell_9/BiasAddBiasAdd5simple_rnn_5/while/simple_rnn_cell_9/MatMul:product:0Csimple_rnn_5/while/simple_rnn_cell_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22.
,simple_rnn_5/while/simple_rnn_cell_9/BiasAdd?
<simple_rnn_5/while/simple_rnn_cell_9/MatMul_1/ReadVariableOpReadVariableOpGsimple_rnn_5_while_simple_rnn_cell_9_matmul_1_readvariableop_resource_0*
_output_shapes

:22*
dtype02>
<simple_rnn_5/while/simple_rnn_cell_9/MatMul_1/ReadVariableOp?
-simple_rnn_5/while/simple_rnn_cell_9/MatMul_1	MLCMatMul simple_rnn_5_while_placeholder_2Dsimple_rnn_5/while/simple_rnn_cell_9/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22/
-simple_rnn_5/while/simple_rnn_cell_9/MatMul_1?
(simple_rnn_5/while/simple_rnn_cell_9/addAddV25simple_rnn_5/while/simple_rnn_cell_9/BiasAdd:output:07simple_rnn_5/while/simple_rnn_cell_9/MatMul_1:product:0*
T0*'
_output_shapes
:?????????22*
(simple_rnn_5/while/simple_rnn_cell_9/add?
)simple_rnn_5/while/simple_rnn_cell_9/TanhTanh,simple_rnn_5/while/simple_rnn_cell_9/add:z:0*
T0*'
_output_shapes
:?????????22+
)simple_rnn_5/while/simple_rnn_cell_9/Tanh?
7simple_rnn_5/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem simple_rnn_5_while_placeholder_1simple_rnn_5_while_placeholder-simple_rnn_5/while/simple_rnn_cell_9/Tanh:y:0*
_output_shapes
: *
element_dtype029
7simple_rnn_5/while/TensorArrayV2Write/TensorListSetItemv
simple_rnn_5/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
simple_rnn_5/while/add/y?
simple_rnn_5/while/addAddV2simple_rnn_5_while_placeholder!simple_rnn_5/while/add/y:output:0*
T0*
_output_shapes
: 2
simple_rnn_5/while/addz
simple_rnn_5/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
simple_rnn_5/while/add_1/y?
simple_rnn_5/while/add_1AddV22simple_rnn_5_while_simple_rnn_5_while_loop_counter#simple_rnn_5/while/add_1/y:output:0*
T0*
_output_shapes
: 2
simple_rnn_5/while/add_1?
simple_rnn_5/while/IdentityIdentitysimple_rnn_5/while/add_1:z:0<^simple_rnn_5/while/simple_rnn_cell_9/BiasAdd/ReadVariableOp;^simple_rnn_5/while/simple_rnn_cell_9/MatMul/ReadVariableOp=^simple_rnn_5/while/simple_rnn_cell_9/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
simple_rnn_5/while/Identity?
simple_rnn_5/while/Identity_1Identity8simple_rnn_5_while_simple_rnn_5_while_maximum_iterations<^simple_rnn_5/while/simple_rnn_cell_9/BiasAdd/ReadVariableOp;^simple_rnn_5/while/simple_rnn_cell_9/MatMul/ReadVariableOp=^simple_rnn_5/while/simple_rnn_cell_9/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
simple_rnn_5/while/Identity_1?
simple_rnn_5/while/Identity_2Identitysimple_rnn_5/while/add:z:0<^simple_rnn_5/while/simple_rnn_cell_9/BiasAdd/ReadVariableOp;^simple_rnn_5/while/simple_rnn_cell_9/MatMul/ReadVariableOp=^simple_rnn_5/while/simple_rnn_cell_9/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
simple_rnn_5/while/Identity_2?
simple_rnn_5/while/Identity_3IdentityGsimple_rnn_5/while/TensorArrayV2Write/TensorListSetItem:output_handle:0<^simple_rnn_5/while/simple_rnn_cell_9/BiasAdd/ReadVariableOp;^simple_rnn_5/while/simple_rnn_cell_9/MatMul/ReadVariableOp=^simple_rnn_5/while/simple_rnn_cell_9/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
simple_rnn_5/while/Identity_3?
simple_rnn_5/while/Identity_4Identity-simple_rnn_5/while/simple_rnn_cell_9/Tanh:y:0<^simple_rnn_5/while/simple_rnn_cell_9/BiasAdd/ReadVariableOp;^simple_rnn_5/while/simple_rnn_cell_9/MatMul/ReadVariableOp=^simple_rnn_5/while/simple_rnn_cell_9/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????22
simple_rnn_5/while/Identity_4"C
simple_rnn_5_while_identity$simple_rnn_5/while/Identity:output:0"G
simple_rnn_5_while_identity_1&simple_rnn_5/while/Identity_1:output:0"G
simple_rnn_5_while_identity_2&simple_rnn_5/while/Identity_2:output:0"G
simple_rnn_5_while_identity_3&simple_rnn_5/while/Identity_3:output:0"G
simple_rnn_5_while_identity_4&simple_rnn_5/while/Identity_4:output:0"d
/simple_rnn_5_while_simple_rnn_5_strided_slice_11simple_rnn_5_while_simple_rnn_5_strided_slice_1_0"?
Dsimple_rnn_5_while_simple_rnn_cell_9_biasadd_readvariableop_resourceFsimple_rnn_5_while_simple_rnn_cell_9_biasadd_readvariableop_resource_0"?
Esimple_rnn_5_while_simple_rnn_cell_9_matmul_1_readvariableop_resourceGsimple_rnn_5_while_simple_rnn_cell_9_matmul_1_readvariableop_resource_0"?
Csimple_rnn_5_while_simple_rnn_cell_9_matmul_readvariableop_resourceEsimple_rnn_5_while_simple_rnn_cell_9_matmul_readvariableop_resource_0"?
ksimple_rnn_5_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_5_tensorarrayunstack_tensorlistfromtensormsimple_rnn_5_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_5_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :?????????2: : :::2z
;simple_rnn_5/while/simple_rnn_cell_9/BiasAdd/ReadVariableOp;simple_rnn_5/while/simple_rnn_cell_9/BiasAdd/ReadVariableOp2x
:simple_rnn_5/while/simple_rnn_cell_9/MatMul/ReadVariableOp:simple_rnn_5/while/simple_rnn_cell_9/MatMul/ReadVariableOp2|
<simple_rnn_5/while/simple_rnn_cell_9/MatMul_1/ReadVariableOp<simple_rnn_5/while/simple_rnn_cell_9/MatMul_1/ReadVariableOp: 
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
?3
?
while_body_101704
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0<
8while_simple_rnn_cell_9_matmul_readvariableop_resource_0=
9while_simple_rnn_cell_9_biasadd_readvariableop_resource_0>
:while_simple_rnn_cell_9_matmul_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor:
6while_simple_rnn_cell_9_matmul_readvariableop_resource;
7while_simple_rnn_cell_9_biasadd_readvariableop_resource<
8while_simple_rnn_cell_9_matmul_1_readvariableop_resource??.while/simple_rnn_cell_9/BiasAdd/ReadVariableOp?-while/simple_rnn_cell_9/MatMul/ReadVariableOp?/while/simple_rnn_cell_9/MatMul_1/ReadVariableOp?
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
-while/simple_rnn_cell_9/MatMul/ReadVariableOpReadVariableOp8while_simple_rnn_cell_9_matmul_readvariableop_resource_0*
_output_shapes
:	?2*
dtype02/
-while/simple_rnn_cell_9/MatMul/ReadVariableOp?
while/simple_rnn_cell_9/MatMul	MLCMatMul0while/TensorArrayV2Read/TensorListGetItem:item:05while/simple_rnn_cell_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22 
while/simple_rnn_cell_9/MatMul?
.while/simple_rnn_cell_9/BiasAdd/ReadVariableOpReadVariableOp9while_simple_rnn_cell_9_biasadd_readvariableop_resource_0*
_output_shapes
:2*
dtype020
.while/simple_rnn_cell_9/BiasAdd/ReadVariableOp?
while/simple_rnn_cell_9/BiasAddBiasAdd(while/simple_rnn_cell_9/MatMul:product:06while/simple_rnn_cell_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22!
while/simple_rnn_cell_9/BiasAdd?
/while/simple_rnn_cell_9/MatMul_1/ReadVariableOpReadVariableOp:while_simple_rnn_cell_9_matmul_1_readvariableop_resource_0*
_output_shapes

:22*
dtype021
/while/simple_rnn_cell_9/MatMul_1/ReadVariableOp?
 while/simple_rnn_cell_9/MatMul_1	MLCMatMulwhile_placeholder_27while/simple_rnn_cell_9/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22"
 while/simple_rnn_cell_9/MatMul_1?
while/simple_rnn_cell_9/addAddV2(while/simple_rnn_cell_9/BiasAdd:output:0*while/simple_rnn_cell_9/MatMul_1:product:0*
T0*'
_output_shapes
:?????????22
while/simple_rnn_cell_9/add?
while/simple_rnn_cell_9/TanhTanhwhile/simple_rnn_cell_9/add:z:0*
T0*'
_output_shapes
:?????????22
while/simple_rnn_cell_9/Tanh?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder while/simple_rnn_cell_9/Tanh:y:0*
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
while/IdentityIdentitywhile/add_1:z:0/^while/simple_rnn_cell_9/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_9/MatMul/ReadVariableOp0^while/simple_rnn_cell_9/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations/^while/simple_rnn_cell_9/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_9/MatMul/ReadVariableOp0^while/simple_rnn_cell_9/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0/^while/simple_rnn_cell_9/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_9/MatMul/ReadVariableOp0^while/simple_rnn_cell_9/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0/^while/simple_rnn_cell_9/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_9/MatMul/ReadVariableOp0^while/simple_rnn_cell_9/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity while/simple_rnn_cell_9/Tanh:y:0/^while/simple_rnn_cell_9/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_9/MatMul/ReadVariableOp0^while/simple_rnn_cell_9/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????22
while/Identity_4")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"t
7while_simple_rnn_cell_9_biasadd_readvariableop_resource9while_simple_rnn_cell_9_biasadd_readvariableop_resource_0"v
8while_simple_rnn_cell_9_matmul_1_readvariableop_resource:while_simple_rnn_cell_9_matmul_1_readvariableop_resource_0"r
6while_simple_rnn_cell_9_matmul_readvariableop_resource8while_simple_rnn_cell_9_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :?????????2: : :::2`
.while/simple_rnn_cell_9/BiasAdd/ReadVariableOp.while/simple_rnn_cell_9/BiasAdd/ReadVariableOp2^
-while/simple_rnn_cell_9/MatMul/ReadVariableOp-while/simple_rnn_cell_9/MatMul/ReadVariableOp2b
/while/simple_rnn_cell_9/MatMul_1/ReadVariableOp/while/simple_rnn_cell_9/MatMul_1/ReadVariableOp: 
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
2__inference_simple_rnn_cell_9_layer_call_fn_102314

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
GPU 2J 8? *U
fPRN
L__inference_simple_rnn_cell_9_layer_call_and_return_conditional_losses_993842
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
?
?
-__inference_simple_rnn_4_layer_call_fn_101624
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
GPU 2J 8? *P
fKRI
G__inference_simple_rnn_4_layer_call_and_return_conditional_losses_993262
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
?
a
E__inference_dropout_5_layer_call_and_return_conditional_losses_100419

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
?
F
*__inference_dropout_4_layer_call_fn_101653

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
:?????????H?* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_4_layer_call_and_return_conditional_losses_1001262
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:?????????H?2

Identity"
identityIdentity:output:0*+
_input_shapes
:?????????H?:T P
,
_output_shapes
:?????????H?
 
_user_specified_nameinputs
?G
?
H__inference_simple_rnn_4_layer_call_and_return_conditional_losses_101356

inputs4
0simple_rnn_cell_8_matmul_readvariableop_resource5
1simple_rnn_cell_8_biasadd_readvariableop_resource6
2simple_rnn_cell_8_matmul_1_readvariableop_resource
identity??(simple_rnn_cell_8/BiasAdd/ReadVariableOp?'simple_rnn_cell_8/MatMul/ReadVariableOp?)simple_rnn_cell_8/MatMul_1/ReadVariableOp?whileD
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
:H?????????2
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
'simple_rnn_cell_8/MatMul/ReadVariableOpReadVariableOp0simple_rnn_cell_8_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02)
'simple_rnn_cell_8/MatMul/ReadVariableOp?
simple_rnn_cell_8/MatMul	MLCMatMulstrided_slice_2:output:0/simple_rnn_cell_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
simple_rnn_cell_8/MatMul?
(simple_rnn_cell_8/BiasAdd/ReadVariableOpReadVariableOp1simple_rnn_cell_8_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02*
(simple_rnn_cell_8/BiasAdd/ReadVariableOp?
simple_rnn_cell_8/BiasAddBiasAdd"simple_rnn_cell_8/MatMul:product:00simple_rnn_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
simple_rnn_cell_8/BiasAdd?
)simple_rnn_cell_8/MatMul_1/ReadVariableOpReadVariableOp2simple_rnn_cell_8_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02+
)simple_rnn_cell_8/MatMul_1/ReadVariableOp?
simple_rnn_cell_8/MatMul_1	MLCMatMulzeros:output:01simple_rnn_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
simple_rnn_cell_8/MatMul_1?
simple_rnn_cell_8/addAddV2"simple_rnn_cell_8/BiasAdd:output:0$simple_rnn_cell_8/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
simple_rnn_cell_8/add?
simple_rnn_cell_8/TanhTanhsimple_rnn_cell_8/add:z:0*
T0*(
_output_shapes
:??????????2
simple_rnn_cell_8/Tanh?
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:00simple_rnn_cell_8_matmul_readvariableop_resource1simple_rnn_cell_8_biasadd_readvariableop_resource2simple_rnn_cell_8_matmul_1_readvariableop_resource*
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
while_body_101290*
condR
while_cond_101289*9
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
:H??????????*
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
:?????????H?2
transpose_1?
IdentityIdentitytranspose_1:y:0)^simple_rnn_cell_8/BiasAdd/ReadVariableOp(^simple_rnn_cell_8/MatMul/ReadVariableOp*^simple_rnn_cell_8/MatMul_1/ReadVariableOp^while*
T0*,
_output_shapes
:?????????H?2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????H:::2T
(simple_rnn_cell_8/BiasAdd/ReadVariableOp(simple_rnn_cell_8/BiasAdd/ReadVariableOp2R
'simple_rnn_cell_8/MatMul/ReadVariableOp'simple_rnn_cell_8/MatMul/ReadVariableOp2V
)simple_rnn_cell_8/MatMul_1/ReadVariableOp)simple_rnn_cell_8/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:?????????H
 
_user_specified_nameinputs
?#
?
while_body_99775
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0#
while_simple_rnn_cell_9_99797_0#
while_simple_rnn_cell_9_99799_0#
while_simple_rnn_cell_9_99801_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor!
while_simple_rnn_cell_9_99797!
while_simple_rnn_cell_9_99799!
while_simple_rnn_cell_9_99801??/while/simple_rnn_cell_9/StatefulPartitionedCall?
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
/while/simple_rnn_cell_9/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_simple_rnn_cell_9_99797_0while_simple_rnn_cell_9_99799_0while_simple_rnn_cell_9_99801_0*
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
GPU 2J 8? *U
fPRN
L__inference_simple_rnn_cell_9_layer_call_and_return_conditional_losses_9940121
/while/simple_rnn_cell_9/StatefulPartitionedCall?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder8while/simple_rnn_cell_9/StatefulPartitionedCall:output:0*
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
while/IdentityIdentitywhile/add_1:z:00^while/simple_rnn_cell_9/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations0^while/simple_rnn_cell_9/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:00^while/simple_rnn_cell_9/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:00^while/simple_rnn_cell_9/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity8while/simple_rnn_cell_9/StatefulPartitionedCall:output:10^while/simple_rnn_cell_9/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????22
while/Identity_4")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"@
while_simple_rnn_cell_9_99797while_simple_rnn_cell_9_99797_0"@
while_simple_rnn_cell_9_99799while_simple_rnn_cell_9_99799_0"@
while_simple_rnn_cell_9_99801while_simple_rnn_cell_9_99801_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :?????????2: : :::2b
/while/simple_rnn_cell_9/StatefulPartitionedCall/while/simple_rnn_cell_9/StatefulPartitionedCall: 
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
2__inference_simple_rnn_cell_8_layer_call_fn_102266

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
GPU 2J 8? *U
fPRN
L__inference_simple_rnn_cell_8_layer_call_and_return_conditional_losses_988892
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
?3
?
while_body_100189
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0<
8while_simple_rnn_cell_9_matmul_readvariableop_resource_0=
9while_simple_rnn_cell_9_biasadd_readvariableop_resource_0>
:while_simple_rnn_cell_9_matmul_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor:
6while_simple_rnn_cell_9_matmul_readvariableop_resource;
7while_simple_rnn_cell_9_biasadd_readvariableop_resource<
8while_simple_rnn_cell_9_matmul_1_readvariableop_resource??.while/simple_rnn_cell_9/BiasAdd/ReadVariableOp?-while/simple_rnn_cell_9/MatMul/ReadVariableOp?/while/simple_rnn_cell_9/MatMul_1/ReadVariableOp?
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
-while/simple_rnn_cell_9/MatMul/ReadVariableOpReadVariableOp8while_simple_rnn_cell_9_matmul_readvariableop_resource_0*
_output_shapes
:	?2*
dtype02/
-while/simple_rnn_cell_9/MatMul/ReadVariableOp?
while/simple_rnn_cell_9/MatMul	MLCMatMul0while/TensorArrayV2Read/TensorListGetItem:item:05while/simple_rnn_cell_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22 
while/simple_rnn_cell_9/MatMul?
.while/simple_rnn_cell_9/BiasAdd/ReadVariableOpReadVariableOp9while_simple_rnn_cell_9_biasadd_readvariableop_resource_0*
_output_shapes
:2*
dtype020
.while/simple_rnn_cell_9/BiasAdd/ReadVariableOp?
while/simple_rnn_cell_9/BiasAddBiasAdd(while/simple_rnn_cell_9/MatMul:product:06while/simple_rnn_cell_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22!
while/simple_rnn_cell_9/BiasAdd?
/while/simple_rnn_cell_9/MatMul_1/ReadVariableOpReadVariableOp:while_simple_rnn_cell_9_matmul_1_readvariableop_resource_0*
_output_shapes

:22*
dtype021
/while/simple_rnn_cell_9/MatMul_1/ReadVariableOp?
 while/simple_rnn_cell_9/MatMul_1	MLCMatMulwhile_placeholder_27while/simple_rnn_cell_9/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22"
 while/simple_rnn_cell_9/MatMul_1?
while/simple_rnn_cell_9/addAddV2(while/simple_rnn_cell_9/BiasAdd:output:0*while/simple_rnn_cell_9/MatMul_1:product:0*
T0*'
_output_shapes
:?????????22
while/simple_rnn_cell_9/add?
while/simple_rnn_cell_9/TanhTanhwhile/simple_rnn_cell_9/add:z:0*
T0*'
_output_shapes
:?????????22
while/simple_rnn_cell_9/Tanh?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder while/simple_rnn_cell_9/Tanh:y:0*
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
while/IdentityIdentitywhile/add_1:z:0/^while/simple_rnn_cell_9/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_9/MatMul/ReadVariableOp0^while/simple_rnn_cell_9/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations/^while/simple_rnn_cell_9/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_9/MatMul/ReadVariableOp0^while/simple_rnn_cell_9/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0/^while/simple_rnn_cell_9/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_9/MatMul/ReadVariableOp0^while/simple_rnn_cell_9/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0/^while/simple_rnn_cell_9/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_9/MatMul/ReadVariableOp0^while/simple_rnn_cell_9/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity while/simple_rnn_cell_9/Tanh:y:0/^while/simple_rnn_cell_9/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_9/MatMul/ReadVariableOp0^while/simple_rnn_cell_9/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????22
while/Identity_4")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"t
7while_simple_rnn_cell_9_biasadd_readvariableop_resource9while_simple_rnn_cell_9_biasadd_readvariableop_resource_0"v
8while_simple_rnn_cell_9_matmul_1_readvariableop_resource:while_simple_rnn_cell_9_matmul_1_readvariableop_resource_0"r
6while_simple_rnn_cell_9_matmul_readvariableop_resource8while_simple_rnn_cell_9_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :?????????2: : :::2`
.while/simple_rnn_cell_9/BiasAdd/ReadVariableOp.while/simple_rnn_cell_9/BiasAdd/ReadVariableOp2^
-while/simple_rnn_cell_9/MatMul/ReadVariableOp-while/simple_rnn_cell_9/MatMul/ReadVariableOp2b
/while/simple_rnn_cell_9/MatMul_1/ReadVariableOp/while/simple_rnn_cell_9/MatMul_1/ReadVariableOp: 
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
?
while_cond_100300
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_100300___redundant_placeholder04
0while_while_cond_100300___redundant_placeholder14
0while_while_cond_100300___redundant_placeholder24
0while_while_cond_100300___redundant_placeholder3
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
?<
?
G__inference_simple_rnn_5_layer_call_and_return_conditional_losses_99838

inputs
simple_rnn_cell_9_99763
simple_rnn_cell_9_99765
simple_rnn_cell_9_99767
identity??)simple_rnn_cell_9/StatefulPartitionedCall?whileD
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
)simple_rnn_cell_9/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0simple_rnn_cell_9_99763simple_rnn_cell_9_99765simple_rnn_cell_9_99767*
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
GPU 2J 8? *U
fPRN
L__inference_simple_rnn_cell_9_layer_call_and_return_conditional_losses_994012+
)simple_rnn_cell_9/StatefulPartitionedCall?
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0simple_rnn_cell_9_99763simple_rnn_cell_9_99765simple_rnn_cell_9_99767*
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
bodyR
while_body_99775*
condR
while_cond_99774*8
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
IdentityIdentitystrided_slice_3:output:0*^simple_rnn_cell_9/StatefulPartitionedCall^while*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*@
_input_shapes/
-:???????????????????:::2V
)simple_rnn_cell_9/StatefulPartitionedCall)simple_rnn_cell_9/StatefulPartitionedCall2
whilewhile:] Y
5
_output_shapes#
!:???????????????????
 
_user_specified_nameinputs
?
?
-__inference_simple_rnn_4_layer_call_fn_101613
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
GPU 2J 8? *P
fKRI
G__inference_simple_rnn_4_layer_call_and_return_conditional_losses_992092
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
?
?
.__inference_sequential_12_layer_call_fn_101111

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
:?????????$**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_sequential_12_layer_call_and_return_conditional_losses_1005222
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????$2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:?????????H::::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????H
 
_user_specified_nameinputs
?
a
E__inference_dropout_4_layer_call_and_return_conditional_losses_101643

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
:?????????H?2
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
:?????????H?2	
dropouti
IdentityIdentitydropout:output:0*
T0*,
_output_shapes
:?????????H?2

Identity"
identityIdentity:output:0*+
_input_shapes
:?????????H?:T P
,
_output_shapes
:?????????H?
 
_user_specified_nameinputs
?3
?
while_body_101424
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0<
8while_simple_rnn_cell_8_matmul_readvariableop_resource_0=
9while_simple_rnn_cell_8_biasadd_readvariableop_resource_0>
:while_simple_rnn_cell_8_matmul_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor:
6while_simple_rnn_cell_8_matmul_readvariableop_resource;
7while_simple_rnn_cell_8_biasadd_readvariableop_resource<
8while_simple_rnn_cell_8_matmul_1_readvariableop_resource??.while/simple_rnn_cell_8/BiasAdd/ReadVariableOp?-while/simple_rnn_cell_8/MatMul/ReadVariableOp?/while/simple_rnn_cell_8/MatMul_1/ReadVariableOp?
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
-while/simple_rnn_cell_8/MatMul/ReadVariableOpReadVariableOp8while_simple_rnn_cell_8_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype02/
-while/simple_rnn_cell_8/MatMul/ReadVariableOp?
while/simple_rnn_cell_8/MatMul	MLCMatMul0while/TensorArrayV2Read/TensorListGetItem:item:05while/simple_rnn_cell_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2 
while/simple_rnn_cell_8/MatMul?
.while/simple_rnn_cell_8/BiasAdd/ReadVariableOpReadVariableOp9while_simple_rnn_cell_8_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype020
.while/simple_rnn_cell_8/BiasAdd/ReadVariableOp?
while/simple_rnn_cell_8/BiasAddBiasAdd(while/simple_rnn_cell_8/MatMul:product:06while/simple_rnn_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2!
while/simple_rnn_cell_8/BiasAdd?
/while/simple_rnn_cell_8/MatMul_1/ReadVariableOpReadVariableOp:while_simple_rnn_cell_8_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype021
/while/simple_rnn_cell_8/MatMul_1/ReadVariableOp?
 while/simple_rnn_cell_8/MatMul_1	MLCMatMulwhile_placeholder_27while/simple_rnn_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2"
 while/simple_rnn_cell_8/MatMul_1?
while/simple_rnn_cell_8/addAddV2(while/simple_rnn_cell_8/BiasAdd:output:0*while/simple_rnn_cell_8/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
while/simple_rnn_cell_8/add?
while/simple_rnn_cell_8/TanhTanhwhile/simple_rnn_cell_8/add:z:0*
T0*(
_output_shapes
:??????????2
while/simple_rnn_cell_8/Tanh?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder while/simple_rnn_cell_8/Tanh:y:0*
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
while/IdentityIdentitywhile/add_1:z:0/^while/simple_rnn_cell_8/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_8/MatMul/ReadVariableOp0^while/simple_rnn_cell_8/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations/^while/simple_rnn_cell_8/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_8/MatMul/ReadVariableOp0^while/simple_rnn_cell_8/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0/^while/simple_rnn_cell_8/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_8/MatMul/ReadVariableOp0^while/simple_rnn_cell_8/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0/^while/simple_rnn_cell_8/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_8/MatMul/ReadVariableOp0^while/simple_rnn_cell_8/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity while/simple_rnn_cell_8/Tanh:y:0/^while/simple_rnn_cell_8/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_8/MatMul/ReadVariableOp0^while/simple_rnn_cell_8/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2
while/Identity_4")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"t
7while_simple_rnn_cell_8_biasadd_readvariableop_resource9while_simple_rnn_cell_8_biasadd_readvariableop_resource_0"v
8while_simple_rnn_cell_8_matmul_1_readvariableop_resource:while_simple_rnn_cell_8_matmul_1_readvariableop_resource_0"r
6while_simple_rnn_cell_8_matmul_readvariableop_resource8while_simple_rnn_cell_8_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*?
_input_shapes.
,: : : : :??????????: : :::2`
.while/simple_rnn_cell_8/BiasAdd/ReadVariableOp.while/simple_rnn_cell_8/BiasAdd/ReadVariableOp2^
-while/simple_rnn_cell_8/MatMul/ReadVariableOp-while/simple_rnn_cell_8/MatMul/ReadVariableOp2b
/while/simple_rnn_cell_8/MatMul_1/ReadVariableOp/while/simple_rnn_cell_8/MatMul_1/ReadVariableOp: 
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
*__inference_dropout_5_layer_call_fn_102184

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
E__inference_dropout_5_layer_call_and_return_conditional_losses_1004242
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
?
e
I__inference_activation_11_layer_call_and_return_conditional_losses_102155

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
?H
?
H__inference_simple_rnn_5_layer_call_and_return_conditional_losses_101770
inputs_04
0simple_rnn_cell_9_matmul_readvariableop_resource5
1simple_rnn_cell_9_biasadd_readvariableop_resource6
2simple_rnn_cell_9_matmul_1_readvariableop_resource
identity??(simple_rnn_cell_9/BiasAdd/ReadVariableOp?'simple_rnn_cell_9/MatMul/ReadVariableOp?)simple_rnn_cell_9/MatMul_1/ReadVariableOp?whileF
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
'simple_rnn_cell_9/MatMul/ReadVariableOpReadVariableOp0simple_rnn_cell_9_matmul_readvariableop_resource*
_output_shapes
:	?2*
dtype02)
'simple_rnn_cell_9/MatMul/ReadVariableOp?
simple_rnn_cell_9/MatMul	MLCMatMulstrided_slice_2:output:0/simple_rnn_cell_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
simple_rnn_cell_9/MatMul?
(simple_rnn_cell_9/BiasAdd/ReadVariableOpReadVariableOp1simple_rnn_cell_9_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02*
(simple_rnn_cell_9/BiasAdd/ReadVariableOp?
simple_rnn_cell_9/BiasAddBiasAdd"simple_rnn_cell_9/MatMul:product:00simple_rnn_cell_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
simple_rnn_cell_9/BiasAdd?
)simple_rnn_cell_9/MatMul_1/ReadVariableOpReadVariableOp2simple_rnn_cell_9_matmul_1_readvariableop_resource*
_output_shapes

:22*
dtype02+
)simple_rnn_cell_9/MatMul_1/ReadVariableOp?
simple_rnn_cell_9/MatMul_1	MLCMatMulzeros:output:01simple_rnn_cell_9/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
simple_rnn_cell_9/MatMul_1?
simple_rnn_cell_9/addAddV2"simple_rnn_cell_9/BiasAdd:output:0$simple_rnn_cell_9/MatMul_1:product:0*
T0*'
_output_shapes
:?????????22
simple_rnn_cell_9/add?
simple_rnn_cell_9/TanhTanhsimple_rnn_cell_9/add:z:0*
T0*'
_output_shapes
:?????????22
simple_rnn_cell_9/Tanh?
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:00simple_rnn_cell_9_matmul_readvariableop_resource1simple_rnn_cell_9_biasadd_readvariableop_resource2simple_rnn_cell_9_matmul_1_readvariableop_resource*
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
while_body_101704*
condR
while_cond_101703*8
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
IdentityIdentitystrided_slice_3:output:0)^simple_rnn_cell_9/BiasAdd/ReadVariableOp(^simple_rnn_cell_9/MatMul/ReadVariableOp*^simple_rnn_cell_9/MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*@
_input_shapes/
-:???????????????????:::2T
(simple_rnn_cell_9/BiasAdd/ReadVariableOp(simple_rnn_cell_9/BiasAdd/ReadVariableOp2R
'simple_rnn_cell_9/MatMul/ReadVariableOp'simple_rnn_cell_9/MatMul/ReadVariableOp2V
)simple_rnn_cell_9/MatMul_1/ReadVariableOp)simple_rnn_cell_9/MatMul_1/ReadVariableOp2
whilewhile:_ [
5
_output_shapes#
!:???????????????????
"
_user_specified_name
inputs/0
?H
?
H__inference_simple_rnn_4_layer_call_and_return_conditional_losses_101602
inputs_04
0simple_rnn_cell_8_matmul_readvariableop_resource5
1simple_rnn_cell_8_biasadd_readvariableop_resource6
2simple_rnn_cell_8_matmul_1_readvariableop_resource
identity??(simple_rnn_cell_8/BiasAdd/ReadVariableOp?'simple_rnn_cell_8/MatMul/ReadVariableOp?)simple_rnn_cell_8/MatMul_1/ReadVariableOp?whileF
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
'simple_rnn_cell_8/MatMul/ReadVariableOpReadVariableOp0simple_rnn_cell_8_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02)
'simple_rnn_cell_8/MatMul/ReadVariableOp?
simple_rnn_cell_8/MatMul	MLCMatMulstrided_slice_2:output:0/simple_rnn_cell_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
simple_rnn_cell_8/MatMul?
(simple_rnn_cell_8/BiasAdd/ReadVariableOpReadVariableOp1simple_rnn_cell_8_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02*
(simple_rnn_cell_8/BiasAdd/ReadVariableOp?
simple_rnn_cell_8/BiasAddBiasAdd"simple_rnn_cell_8/MatMul:product:00simple_rnn_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
simple_rnn_cell_8/BiasAdd?
)simple_rnn_cell_8/MatMul_1/ReadVariableOpReadVariableOp2simple_rnn_cell_8_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02+
)simple_rnn_cell_8/MatMul_1/ReadVariableOp?
simple_rnn_cell_8/MatMul_1	MLCMatMulzeros:output:01simple_rnn_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
simple_rnn_cell_8/MatMul_1?
simple_rnn_cell_8/addAddV2"simple_rnn_cell_8/BiasAdd:output:0$simple_rnn_cell_8/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
simple_rnn_cell_8/add?
simple_rnn_cell_8/TanhTanhsimple_rnn_cell_8/add:z:0*
T0*(
_output_shapes
:??????????2
simple_rnn_cell_8/Tanh?
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:00simple_rnn_cell_8_matmul_readvariableop_resource1simple_rnn_cell_8_biasadd_readvariableop_resource2simple_rnn_cell_8_matmul_1_readvariableop_resource*
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
while_body_101536*
condR
while_cond_101535*9
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
IdentityIdentitytranspose_1:y:0)^simple_rnn_cell_8/BiasAdd/ReadVariableOp(^simple_rnn_cell_8/MatMul/ReadVariableOp*^simple_rnn_cell_8/MatMul_1/ReadVariableOp^while*
T0*5
_output_shapes#
!:???????????????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????:::2T
(simple_rnn_cell_8/BiasAdd/ReadVariableOp(simple_rnn_cell_8/BiasAdd/ReadVariableOp2R
'simple_rnn_cell_8/MatMul/ReadVariableOp'simple_rnn_cell_8/MatMul/ReadVariableOp2V
)simple_rnn_cell_8/MatMul_1/ReadVariableOp)simple_rnn_cell_8/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :??????????????????
"
_user_specified_name
inputs/0
?G
?
H__inference_simple_rnn_5_layer_call_and_return_conditional_losses_102016

inputs4
0simple_rnn_cell_9_matmul_readvariableop_resource5
1simple_rnn_cell_9_biasadd_readvariableop_resource6
2simple_rnn_cell_9_matmul_1_readvariableop_resource
identity??(simple_rnn_cell_9/BiasAdd/ReadVariableOp?'simple_rnn_cell_9/MatMul/ReadVariableOp?)simple_rnn_cell_9/MatMul_1/ReadVariableOp?whileD
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
:H??????????2
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
'simple_rnn_cell_9/MatMul/ReadVariableOpReadVariableOp0simple_rnn_cell_9_matmul_readvariableop_resource*
_output_shapes
:	?2*
dtype02)
'simple_rnn_cell_9/MatMul/ReadVariableOp?
simple_rnn_cell_9/MatMul	MLCMatMulstrided_slice_2:output:0/simple_rnn_cell_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
simple_rnn_cell_9/MatMul?
(simple_rnn_cell_9/BiasAdd/ReadVariableOpReadVariableOp1simple_rnn_cell_9_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02*
(simple_rnn_cell_9/BiasAdd/ReadVariableOp?
simple_rnn_cell_9/BiasAddBiasAdd"simple_rnn_cell_9/MatMul:product:00simple_rnn_cell_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
simple_rnn_cell_9/BiasAdd?
)simple_rnn_cell_9/MatMul_1/ReadVariableOpReadVariableOp2simple_rnn_cell_9_matmul_1_readvariableop_resource*
_output_shapes

:22*
dtype02+
)simple_rnn_cell_9/MatMul_1/ReadVariableOp?
simple_rnn_cell_9/MatMul_1	MLCMatMulzeros:output:01simple_rnn_cell_9/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
simple_rnn_cell_9/MatMul_1?
simple_rnn_cell_9/addAddV2"simple_rnn_cell_9/BiasAdd:output:0$simple_rnn_cell_9/MatMul_1:product:0*
T0*'
_output_shapes
:?????????22
simple_rnn_cell_9/add?
simple_rnn_cell_9/TanhTanhsimple_rnn_cell_9/add:z:0*
T0*'
_output_shapes
:?????????22
simple_rnn_cell_9/Tanh?
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:00simple_rnn_cell_9_matmul_readvariableop_resource1simple_rnn_cell_9_biasadd_readvariableop_resource2simple_rnn_cell_9_matmul_1_readvariableop_resource*
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
while_body_101950*
condR
while_cond_101949*8
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
:H?????????2*
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
:?????????H22
transpose_1?
IdentityIdentitystrided_slice_3:output:0)^simple_rnn_cell_9/BiasAdd/ReadVariableOp(^simple_rnn_cell_9/MatMul/ReadVariableOp*^simple_rnn_cell_9/MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????H?:::2T
(simple_rnn_cell_9/BiasAdd/ReadVariableOp(simple_rnn_cell_9/BiasAdd/ReadVariableOp2R
'simple_rnn_cell_9/MatMul/ReadVariableOp'simple_rnn_cell_9/MatMul/ReadVariableOp2V
)simple_rnn_cell_9/MatMul_1/ReadVariableOp)simple_rnn_cell_9/MatMul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:?????????H?
 
_user_specified_nameinputs
?G
?
H__inference_simple_rnn_5_layer_call_and_return_conditional_losses_100367

inputs4
0simple_rnn_cell_9_matmul_readvariableop_resource5
1simple_rnn_cell_9_biasadd_readvariableop_resource6
2simple_rnn_cell_9_matmul_1_readvariableop_resource
identity??(simple_rnn_cell_9/BiasAdd/ReadVariableOp?'simple_rnn_cell_9/MatMul/ReadVariableOp?)simple_rnn_cell_9/MatMul_1/ReadVariableOp?whileD
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
:H??????????2
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
'simple_rnn_cell_9/MatMul/ReadVariableOpReadVariableOp0simple_rnn_cell_9_matmul_readvariableop_resource*
_output_shapes
:	?2*
dtype02)
'simple_rnn_cell_9/MatMul/ReadVariableOp?
simple_rnn_cell_9/MatMul	MLCMatMulstrided_slice_2:output:0/simple_rnn_cell_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
simple_rnn_cell_9/MatMul?
(simple_rnn_cell_9/BiasAdd/ReadVariableOpReadVariableOp1simple_rnn_cell_9_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02*
(simple_rnn_cell_9/BiasAdd/ReadVariableOp?
simple_rnn_cell_9/BiasAddBiasAdd"simple_rnn_cell_9/MatMul:product:00simple_rnn_cell_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
simple_rnn_cell_9/BiasAdd?
)simple_rnn_cell_9/MatMul_1/ReadVariableOpReadVariableOp2simple_rnn_cell_9_matmul_1_readvariableop_resource*
_output_shapes

:22*
dtype02+
)simple_rnn_cell_9/MatMul_1/ReadVariableOp?
simple_rnn_cell_9/MatMul_1	MLCMatMulzeros:output:01simple_rnn_cell_9/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
simple_rnn_cell_9/MatMul_1?
simple_rnn_cell_9/addAddV2"simple_rnn_cell_9/BiasAdd:output:0$simple_rnn_cell_9/MatMul_1:product:0*
T0*'
_output_shapes
:?????????22
simple_rnn_cell_9/add?
simple_rnn_cell_9/TanhTanhsimple_rnn_cell_9/add:z:0*
T0*'
_output_shapes
:?????????22
simple_rnn_cell_9/Tanh?
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:00simple_rnn_cell_9_matmul_readvariableop_resource1simple_rnn_cell_9_biasadd_readvariableop_resource2simple_rnn_cell_9_matmul_1_readvariableop_resource*
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
while_body_100301*
condR
while_cond_100300*8
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
:H?????????2*
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
:?????????H22
transpose_1?
IdentityIdentitystrided_slice_3:output:0)^simple_rnn_cell_9/BiasAdd/ReadVariableOp(^simple_rnn_cell_9/MatMul/ReadVariableOp*^simple_rnn_cell_9/MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????H?:::2T
(simple_rnn_cell_9/BiasAdd/ReadVariableOp(simple_rnn_cell_9/BiasAdd/ReadVariableOp2R
'simple_rnn_cell_9/MatMul/ReadVariableOp'simple_rnn_cell_9/MatMul/ReadVariableOp2V
)simple_rnn_cell_9/MatMul_1/ReadVariableOp)simple_rnn_cell_9/MatMul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:?????????H?
 
_user_specified_nameinputs
?3
?
while_body_100008
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0<
8while_simple_rnn_cell_8_matmul_readvariableop_resource_0=
9while_simple_rnn_cell_8_biasadd_readvariableop_resource_0>
:while_simple_rnn_cell_8_matmul_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor:
6while_simple_rnn_cell_8_matmul_readvariableop_resource;
7while_simple_rnn_cell_8_biasadd_readvariableop_resource<
8while_simple_rnn_cell_8_matmul_1_readvariableop_resource??.while/simple_rnn_cell_8/BiasAdd/ReadVariableOp?-while/simple_rnn_cell_8/MatMul/ReadVariableOp?/while/simple_rnn_cell_8/MatMul_1/ReadVariableOp?
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
-while/simple_rnn_cell_8/MatMul/ReadVariableOpReadVariableOp8while_simple_rnn_cell_8_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype02/
-while/simple_rnn_cell_8/MatMul/ReadVariableOp?
while/simple_rnn_cell_8/MatMul	MLCMatMul0while/TensorArrayV2Read/TensorListGetItem:item:05while/simple_rnn_cell_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2 
while/simple_rnn_cell_8/MatMul?
.while/simple_rnn_cell_8/BiasAdd/ReadVariableOpReadVariableOp9while_simple_rnn_cell_8_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype020
.while/simple_rnn_cell_8/BiasAdd/ReadVariableOp?
while/simple_rnn_cell_8/BiasAddBiasAdd(while/simple_rnn_cell_8/MatMul:product:06while/simple_rnn_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2!
while/simple_rnn_cell_8/BiasAdd?
/while/simple_rnn_cell_8/MatMul_1/ReadVariableOpReadVariableOp:while_simple_rnn_cell_8_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype021
/while/simple_rnn_cell_8/MatMul_1/ReadVariableOp?
 while/simple_rnn_cell_8/MatMul_1	MLCMatMulwhile_placeholder_27while/simple_rnn_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2"
 while/simple_rnn_cell_8/MatMul_1?
while/simple_rnn_cell_8/addAddV2(while/simple_rnn_cell_8/BiasAdd:output:0*while/simple_rnn_cell_8/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
while/simple_rnn_cell_8/add?
while/simple_rnn_cell_8/TanhTanhwhile/simple_rnn_cell_8/add:z:0*
T0*(
_output_shapes
:??????????2
while/simple_rnn_cell_8/Tanh?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder while/simple_rnn_cell_8/Tanh:y:0*
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
while/IdentityIdentitywhile/add_1:z:0/^while/simple_rnn_cell_8/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_8/MatMul/ReadVariableOp0^while/simple_rnn_cell_8/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations/^while/simple_rnn_cell_8/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_8/MatMul/ReadVariableOp0^while/simple_rnn_cell_8/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0/^while/simple_rnn_cell_8/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_8/MatMul/ReadVariableOp0^while/simple_rnn_cell_8/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0/^while/simple_rnn_cell_8/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_8/MatMul/ReadVariableOp0^while/simple_rnn_cell_8/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity while/simple_rnn_cell_8/Tanh:y:0/^while/simple_rnn_cell_8/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_8/MatMul/ReadVariableOp0^while/simple_rnn_cell_8/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2
while/Identity_4")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"t
7while_simple_rnn_cell_8_biasadd_readvariableop_resource9while_simple_rnn_cell_8_biasadd_readvariableop_resource_0"v
8while_simple_rnn_cell_8_matmul_1_readvariableop_resource:while_simple_rnn_cell_8_matmul_1_readvariableop_resource_0"r
6while_simple_rnn_cell_8_matmul_readvariableop_resource8while_simple_rnn_cell_8_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*?
_input_shapes.
,: : : : :??????????: : :::2`
.while/simple_rnn_cell_8/BiasAdd/ReadVariableOp.while/simple_rnn_cell_8/BiasAdd/ReadVariableOp2^
-while/simple_rnn_cell_8/MatMul/ReadVariableOp-while/simple_rnn_cell_8/MatMul/ReadVariableOp2b
/while/simple_rnn_cell_8/MatMul_1/ReadVariableOp/while/simple_rnn_cell_8/MatMul_1/ReadVariableOp: 
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
?3
?
while_body_100301
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0<
8while_simple_rnn_cell_9_matmul_readvariableop_resource_0=
9while_simple_rnn_cell_9_biasadd_readvariableop_resource_0>
:while_simple_rnn_cell_9_matmul_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor:
6while_simple_rnn_cell_9_matmul_readvariableop_resource;
7while_simple_rnn_cell_9_biasadd_readvariableop_resource<
8while_simple_rnn_cell_9_matmul_1_readvariableop_resource??.while/simple_rnn_cell_9/BiasAdd/ReadVariableOp?-while/simple_rnn_cell_9/MatMul/ReadVariableOp?/while/simple_rnn_cell_9/MatMul_1/ReadVariableOp?
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
-while/simple_rnn_cell_9/MatMul/ReadVariableOpReadVariableOp8while_simple_rnn_cell_9_matmul_readvariableop_resource_0*
_output_shapes
:	?2*
dtype02/
-while/simple_rnn_cell_9/MatMul/ReadVariableOp?
while/simple_rnn_cell_9/MatMul	MLCMatMul0while/TensorArrayV2Read/TensorListGetItem:item:05while/simple_rnn_cell_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22 
while/simple_rnn_cell_9/MatMul?
.while/simple_rnn_cell_9/BiasAdd/ReadVariableOpReadVariableOp9while_simple_rnn_cell_9_biasadd_readvariableop_resource_0*
_output_shapes
:2*
dtype020
.while/simple_rnn_cell_9/BiasAdd/ReadVariableOp?
while/simple_rnn_cell_9/BiasAddBiasAdd(while/simple_rnn_cell_9/MatMul:product:06while/simple_rnn_cell_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22!
while/simple_rnn_cell_9/BiasAdd?
/while/simple_rnn_cell_9/MatMul_1/ReadVariableOpReadVariableOp:while_simple_rnn_cell_9_matmul_1_readvariableop_resource_0*
_output_shapes

:22*
dtype021
/while/simple_rnn_cell_9/MatMul_1/ReadVariableOp?
 while/simple_rnn_cell_9/MatMul_1	MLCMatMulwhile_placeholder_27while/simple_rnn_cell_9/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22"
 while/simple_rnn_cell_9/MatMul_1?
while/simple_rnn_cell_9/addAddV2(while/simple_rnn_cell_9/BiasAdd:output:0*while/simple_rnn_cell_9/MatMul_1:product:0*
T0*'
_output_shapes
:?????????22
while/simple_rnn_cell_9/add?
while/simple_rnn_cell_9/TanhTanhwhile/simple_rnn_cell_9/add:z:0*
T0*'
_output_shapes
:?????????22
while/simple_rnn_cell_9/Tanh?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder while/simple_rnn_cell_9/Tanh:y:0*
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
while/IdentityIdentitywhile/add_1:z:0/^while/simple_rnn_cell_9/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_9/MatMul/ReadVariableOp0^while/simple_rnn_cell_9/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations/^while/simple_rnn_cell_9/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_9/MatMul/ReadVariableOp0^while/simple_rnn_cell_9/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0/^while/simple_rnn_cell_9/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_9/MatMul/ReadVariableOp0^while/simple_rnn_cell_9/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0/^while/simple_rnn_cell_9/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_9/MatMul/ReadVariableOp0^while/simple_rnn_cell_9/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity while/simple_rnn_cell_9/Tanh:y:0/^while/simple_rnn_cell_9/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_9/MatMul/ReadVariableOp0^while/simple_rnn_cell_9/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????22
while/Identity_4")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"t
7while_simple_rnn_cell_9_biasadd_readvariableop_resource9while_simple_rnn_cell_9_biasadd_readvariableop_resource_0"v
8while_simple_rnn_cell_9_matmul_1_readvariableop_resource:while_simple_rnn_cell_9_matmul_1_readvariableop_resource_0"r
6while_simple_rnn_cell_9_matmul_readvariableop_resource8while_simple_rnn_cell_9_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :?????????2: : :::2`
.while/simple_rnn_cell_9/BiasAdd/ReadVariableOp.while/simple_rnn_cell_9/BiasAdd/ReadVariableOp2^
-while/simple_rnn_cell_9/MatMul/ReadVariableOp-while/simple_rnn_cell_9/MatMul/ReadVariableOp2b
/while/simple_rnn_cell_9/MatMul_1/ReadVariableOp/while/simple_rnn_cell_9/MatMul_1/ReadVariableOp: 
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
?
while_cond_100188
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_100188___redundant_placeholder04
0while_while_cond_100188___redundant_placeholder14
0while_while_cond_100188___redundant_placeholder24
0while_while_cond_100188___redundant_placeholder3
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
?<
?
G__inference_simple_rnn_4_layer_call_and_return_conditional_losses_99209

inputs
simple_rnn_cell_8_99134
simple_rnn_cell_8_99136
simple_rnn_cell_8_99138
identity??)simple_rnn_cell_8/StatefulPartitionedCall?whileD
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
)simple_rnn_cell_8/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0simple_rnn_cell_8_99134simple_rnn_cell_8_99136simple_rnn_cell_8_99138*
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
GPU 2J 8? *U
fPRN
L__inference_simple_rnn_cell_8_layer_call_and_return_conditional_losses_988722+
)simple_rnn_cell_8/StatefulPartitionedCall?
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0simple_rnn_cell_8_99134simple_rnn_cell_8_99136simple_rnn_cell_8_99138*
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
bodyR
while_body_99146*
condR
while_cond_99145*9
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
IdentityIdentitytranspose_1:y:0*^simple_rnn_cell_8/StatefulPartitionedCall^while*
T0*5
_output_shapes#
!:???????????????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????:::2V
)simple_rnn_cell_8/StatefulPartitionedCall)simple_rnn_cell_8/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?O
?
__inference__traced_save_102456
file_prefix.
*savev2_dense_14_kernel_read_readvariableop,
(savev2_dense_14_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableopD
@savev2_simple_rnn_4_simple_rnn_cell_8_kernel_read_readvariableopN
Jsavev2_simple_rnn_4_simple_rnn_cell_8_recurrent_kernel_read_readvariableopB
>savev2_simple_rnn_4_simple_rnn_cell_8_bias_read_readvariableopD
@savev2_simple_rnn_5_simple_rnn_cell_9_kernel_read_readvariableopN
Jsavev2_simple_rnn_5_simple_rnn_cell_9_recurrent_kernel_read_readvariableopB
>savev2_simple_rnn_5_simple_rnn_cell_9_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop&
"savev2_total_2_read_readvariableop&
"savev2_count_2_read_readvariableop5
1savev2_adam_dense_14_kernel_m_read_readvariableop3
/savev2_adam_dense_14_bias_m_read_readvariableopK
Gsavev2_adam_simple_rnn_4_simple_rnn_cell_8_kernel_m_read_readvariableopU
Qsavev2_adam_simple_rnn_4_simple_rnn_cell_8_recurrent_kernel_m_read_readvariableopI
Esavev2_adam_simple_rnn_4_simple_rnn_cell_8_bias_m_read_readvariableopK
Gsavev2_adam_simple_rnn_5_simple_rnn_cell_9_kernel_m_read_readvariableopU
Qsavev2_adam_simple_rnn_5_simple_rnn_cell_9_recurrent_kernel_m_read_readvariableopI
Esavev2_adam_simple_rnn_5_simple_rnn_cell_9_bias_m_read_readvariableop5
1savev2_adam_dense_14_kernel_v_read_readvariableop3
/savev2_adam_dense_14_bias_v_read_readvariableopK
Gsavev2_adam_simple_rnn_4_simple_rnn_cell_8_kernel_v_read_readvariableopU
Qsavev2_adam_simple_rnn_4_simple_rnn_cell_8_recurrent_kernel_v_read_readvariableopI
Esavev2_adam_simple_rnn_4_simple_rnn_cell_8_bias_v_read_readvariableopK
Gsavev2_adam_simple_rnn_5_simple_rnn_cell_9_kernel_v_read_readvariableopU
Qsavev2_adam_simple_rnn_5_simple_rnn_cell_9_recurrent_kernel_v_read_readvariableopI
Esavev2_adam_simple_rnn_5_simple_rnn_cell_9_bias_v_read_readvariableop
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
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_14_kernel_read_readvariableop(savev2_dense_14_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop@savev2_simple_rnn_4_simple_rnn_cell_8_kernel_read_readvariableopJsavev2_simple_rnn_4_simple_rnn_cell_8_recurrent_kernel_read_readvariableop>savev2_simple_rnn_4_simple_rnn_cell_8_bias_read_readvariableop@savev2_simple_rnn_5_simple_rnn_cell_9_kernel_read_readvariableopJsavev2_simple_rnn_5_simple_rnn_cell_9_recurrent_kernel_read_readvariableop>savev2_simple_rnn_5_simple_rnn_cell_9_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop"savev2_total_2_read_readvariableop"savev2_count_2_read_readvariableop1savev2_adam_dense_14_kernel_m_read_readvariableop/savev2_adam_dense_14_bias_m_read_readvariableopGsavev2_adam_simple_rnn_4_simple_rnn_cell_8_kernel_m_read_readvariableopQsavev2_adam_simple_rnn_4_simple_rnn_cell_8_recurrent_kernel_m_read_readvariableopEsavev2_adam_simple_rnn_4_simple_rnn_cell_8_bias_m_read_readvariableopGsavev2_adam_simple_rnn_5_simple_rnn_cell_9_kernel_m_read_readvariableopQsavev2_adam_simple_rnn_5_simple_rnn_cell_9_recurrent_kernel_m_read_readvariableopEsavev2_adam_simple_rnn_5_simple_rnn_cell_9_bias_m_read_readvariableop1savev2_adam_dense_14_kernel_v_read_readvariableop/savev2_adam_dense_14_bias_v_read_readvariableopGsavev2_adam_simple_rnn_4_simple_rnn_cell_8_kernel_v_read_readvariableopQsavev2_adam_simple_rnn_4_simple_rnn_cell_8_recurrent_kernel_v_read_readvariableopEsavev2_adam_simple_rnn_4_simple_rnn_cell_8_bias_v_read_readvariableopGsavev2_adam_simple_rnn_5_simple_rnn_cell_9_kernel_v_read_readvariableopQsavev2_adam_simple_rnn_5_simple_rnn_cell_9_recurrent_kernel_v_read_readvariableopEsavev2_adam_simple_rnn_5_simple_rnn_cell_9_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
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
?: :2$:$: : : : : :	?:
??:?:	?2:22:2: : : : : : :2$:$:	?:
??:?:	?2:22:2:2$:$:	?:
??:?:	?2:22:2: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:2$: 

_output_shapes
:$:
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

:2$: 

_output_shapes
:$:%!

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

:2$: 

_output_shapes
:$:%!

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
?
a
E__inference_dropout_5_layer_call_and_return_conditional_losses_102169

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
?G
?
H__inference_simple_rnn_4_layer_call_and_return_conditional_losses_101244

inputs4
0simple_rnn_cell_8_matmul_readvariableop_resource5
1simple_rnn_cell_8_biasadd_readvariableop_resource6
2simple_rnn_cell_8_matmul_1_readvariableop_resource
identity??(simple_rnn_cell_8/BiasAdd/ReadVariableOp?'simple_rnn_cell_8/MatMul/ReadVariableOp?)simple_rnn_cell_8/MatMul_1/ReadVariableOp?whileD
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
:H?????????2
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
'simple_rnn_cell_8/MatMul/ReadVariableOpReadVariableOp0simple_rnn_cell_8_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02)
'simple_rnn_cell_8/MatMul/ReadVariableOp?
simple_rnn_cell_8/MatMul	MLCMatMulstrided_slice_2:output:0/simple_rnn_cell_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
simple_rnn_cell_8/MatMul?
(simple_rnn_cell_8/BiasAdd/ReadVariableOpReadVariableOp1simple_rnn_cell_8_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02*
(simple_rnn_cell_8/BiasAdd/ReadVariableOp?
simple_rnn_cell_8/BiasAddBiasAdd"simple_rnn_cell_8/MatMul:product:00simple_rnn_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
simple_rnn_cell_8/BiasAdd?
)simple_rnn_cell_8/MatMul_1/ReadVariableOpReadVariableOp2simple_rnn_cell_8_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02+
)simple_rnn_cell_8/MatMul_1/ReadVariableOp?
simple_rnn_cell_8/MatMul_1	MLCMatMulzeros:output:01simple_rnn_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
simple_rnn_cell_8/MatMul_1?
simple_rnn_cell_8/addAddV2"simple_rnn_cell_8/BiasAdd:output:0$simple_rnn_cell_8/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
simple_rnn_cell_8/add?
simple_rnn_cell_8/TanhTanhsimple_rnn_cell_8/add:z:0*
T0*(
_output_shapes
:??????????2
simple_rnn_cell_8/Tanh?
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:00simple_rnn_cell_8_matmul_readvariableop_resource1simple_rnn_cell_8_biasadd_readvariableop_resource2simple_rnn_cell_8_matmul_1_readvariableop_resource*
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
while_body_101178*
condR
while_cond_101177*9
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
:H??????????*
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
:?????????H?2
transpose_1?
IdentityIdentitytranspose_1:y:0)^simple_rnn_cell_8/BiasAdd/ReadVariableOp(^simple_rnn_cell_8/MatMul/ReadVariableOp*^simple_rnn_cell_8/MatMul_1/ReadVariableOp^while*
T0*,
_output_shapes
:?????????H?2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????H:::2T
(simple_rnn_cell_8/BiasAdd/ReadVariableOp(simple_rnn_cell_8/BiasAdd/ReadVariableOp2R
'simple_rnn_cell_8/MatMul/ReadVariableOp'simple_rnn_cell_8/MatMul/ReadVariableOp2V
)simple_rnn_cell_8/MatMul_1/ReadVariableOp)simple_rnn_cell_8/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:?????????H
 
_user_specified_nameinputs
??
?	
 __inference__wrapped_model_98823
simple_rnn_4_inputO
Ksequential_12_simple_rnn_4_simple_rnn_cell_8_matmul_readvariableop_resourceP
Lsequential_12_simple_rnn_4_simple_rnn_cell_8_biasadd_readvariableop_resourceQ
Msequential_12_simple_rnn_4_simple_rnn_cell_8_matmul_1_readvariableop_resourceO
Ksequential_12_simple_rnn_5_simple_rnn_cell_9_matmul_readvariableop_resourceP
Lsequential_12_simple_rnn_5_simple_rnn_cell_9_biasadd_readvariableop_resourceQ
Msequential_12_simple_rnn_5_simple_rnn_cell_9_matmul_1_readvariableop_resource<
8sequential_12_dense_14_mlcmatmul_readvariableop_resource:
6sequential_12_dense_14_biasadd_readvariableop_resource
identity??-sequential_12/dense_14/BiasAdd/ReadVariableOp?/sequential_12/dense_14/MLCMatMul/ReadVariableOp?Csequential_12/simple_rnn_4/simple_rnn_cell_8/BiasAdd/ReadVariableOp?Bsequential_12/simple_rnn_4/simple_rnn_cell_8/MatMul/ReadVariableOp?Dsequential_12/simple_rnn_4/simple_rnn_cell_8/MatMul_1/ReadVariableOp? sequential_12/simple_rnn_4/while?Csequential_12/simple_rnn_5/simple_rnn_cell_9/BiasAdd/ReadVariableOp?Bsequential_12/simple_rnn_5/simple_rnn_cell_9/MatMul/ReadVariableOp?Dsequential_12/simple_rnn_5/simple_rnn_cell_9/MatMul_1/ReadVariableOp? sequential_12/simple_rnn_5/while?
 sequential_12/simple_rnn_4/ShapeShapesimple_rnn_4_input*
T0*
_output_shapes
:2"
 sequential_12/simple_rnn_4/Shape?
.sequential_12/simple_rnn_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 20
.sequential_12/simple_rnn_4/strided_slice/stack?
0sequential_12/simple_rnn_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0sequential_12/simple_rnn_4/strided_slice/stack_1?
0sequential_12/simple_rnn_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0sequential_12/simple_rnn_4/strided_slice/stack_2?
(sequential_12/simple_rnn_4/strided_sliceStridedSlice)sequential_12/simple_rnn_4/Shape:output:07sequential_12/simple_rnn_4/strided_slice/stack:output:09sequential_12/simple_rnn_4/strided_slice/stack_1:output:09sequential_12/simple_rnn_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2*
(sequential_12/simple_rnn_4/strided_slice?
&sequential_12/simple_rnn_4/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2(
&sequential_12/simple_rnn_4/zeros/mul/y?
$sequential_12/simple_rnn_4/zeros/mulMul1sequential_12/simple_rnn_4/strided_slice:output:0/sequential_12/simple_rnn_4/zeros/mul/y:output:0*
T0*
_output_shapes
: 2&
$sequential_12/simple_rnn_4/zeros/mul?
'sequential_12/simple_rnn_4/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2)
'sequential_12/simple_rnn_4/zeros/Less/y?
%sequential_12/simple_rnn_4/zeros/LessLess(sequential_12/simple_rnn_4/zeros/mul:z:00sequential_12/simple_rnn_4/zeros/Less/y:output:0*
T0*
_output_shapes
: 2'
%sequential_12/simple_rnn_4/zeros/Less?
)sequential_12/simple_rnn_4/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2+
)sequential_12/simple_rnn_4/zeros/packed/1?
'sequential_12/simple_rnn_4/zeros/packedPack1sequential_12/simple_rnn_4/strided_slice:output:02sequential_12/simple_rnn_4/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2)
'sequential_12/simple_rnn_4/zeros/packed?
&sequential_12/simple_rnn_4/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2(
&sequential_12/simple_rnn_4/zeros/Const?
 sequential_12/simple_rnn_4/zerosFill0sequential_12/simple_rnn_4/zeros/packed:output:0/sequential_12/simple_rnn_4/zeros/Const:output:0*
T0*(
_output_shapes
:??????????2"
 sequential_12/simple_rnn_4/zeros?
)sequential_12/simple_rnn_4/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2+
)sequential_12/simple_rnn_4/transpose/perm?
$sequential_12/simple_rnn_4/transpose	Transposesimple_rnn_4_input2sequential_12/simple_rnn_4/transpose/perm:output:0*
T0*+
_output_shapes
:H?????????2&
$sequential_12/simple_rnn_4/transpose?
"sequential_12/simple_rnn_4/Shape_1Shape(sequential_12/simple_rnn_4/transpose:y:0*
T0*
_output_shapes
:2$
"sequential_12/simple_rnn_4/Shape_1?
0sequential_12/simple_rnn_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0sequential_12/simple_rnn_4/strided_slice_1/stack?
2sequential_12/simple_rnn_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2sequential_12/simple_rnn_4/strided_slice_1/stack_1?
2sequential_12/simple_rnn_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2sequential_12/simple_rnn_4/strided_slice_1/stack_2?
*sequential_12/simple_rnn_4/strided_slice_1StridedSlice+sequential_12/simple_rnn_4/Shape_1:output:09sequential_12/simple_rnn_4/strided_slice_1/stack:output:0;sequential_12/simple_rnn_4/strided_slice_1/stack_1:output:0;sequential_12/simple_rnn_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*sequential_12/simple_rnn_4/strided_slice_1?
6sequential_12/simple_rnn_4/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????28
6sequential_12/simple_rnn_4/TensorArrayV2/element_shape?
(sequential_12/simple_rnn_4/TensorArrayV2TensorListReserve?sequential_12/simple_rnn_4/TensorArrayV2/element_shape:output:03sequential_12/simple_rnn_4/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02*
(sequential_12/simple_rnn_4/TensorArrayV2?
Psequential_12/simple_rnn_4/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2R
Psequential_12/simple_rnn_4/TensorArrayUnstack/TensorListFromTensor/element_shape?
Bsequential_12/simple_rnn_4/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor(sequential_12/simple_rnn_4/transpose:y:0Ysequential_12/simple_rnn_4/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02D
Bsequential_12/simple_rnn_4/TensorArrayUnstack/TensorListFromTensor?
0sequential_12/simple_rnn_4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0sequential_12/simple_rnn_4/strided_slice_2/stack?
2sequential_12/simple_rnn_4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2sequential_12/simple_rnn_4/strided_slice_2/stack_1?
2sequential_12/simple_rnn_4/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2sequential_12/simple_rnn_4/strided_slice_2/stack_2?
*sequential_12/simple_rnn_4/strided_slice_2StridedSlice(sequential_12/simple_rnn_4/transpose:y:09sequential_12/simple_rnn_4/strided_slice_2/stack:output:0;sequential_12/simple_rnn_4/strided_slice_2/stack_1:output:0;sequential_12/simple_rnn_4/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2,
*sequential_12/simple_rnn_4/strided_slice_2?
Bsequential_12/simple_rnn_4/simple_rnn_cell_8/MatMul/ReadVariableOpReadVariableOpKsequential_12_simple_rnn_4_simple_rnn_cell_8_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02D
Bsequential_12/simple_rnn_4/simple_rnn_cell_8/MatMul/ReadVariableOp?
3sequential_12/simple_rnn_4/simple_rnn_cell_8/MatMul	MLCMatMul3sequential_12/simple_rnn_4/strided_slice_2:output:0Jsequential_12/simple_rnn_4/simple_rnn_cell_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????25
3sequential_12/simple_rnn_4/simple_rnn_cell_8/MatMul?
Csequential_12/simple_rnn_4/simple_rnn_cell_8/BiasAdd/ReadVariableOpReadVariableOpLsequential_12_simple_rnn_4_simple_rnn_cell_8_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02E
Csequential_12/simple_rnn_4/simple_rnn_cell_8/BiasAdd/ReadVariableOp?
4sequential_12/simple_rnn_4/simple_rnn_cell_8/BiasAddBiasAdd=sequential_12/simple_rnn_4/simple_rnn_cell_8/MatMul:product:0Ksequential_12/simple_rnn_4/simple_rnn_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????26
4sequential_12/simple_rnn_4/simple_rnn_cell_8/BiasAdd?
Dsequential_12/simple_rnn_4/simple_rnn_cell_8/MatMul_1/ReadVariableOpReadVariableOpMsequential_12_simple_rnn_4_simple_rnn_cell_8_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02F
Dsequential_12/simple_rnn_4/simple_rnn_cell_8/MatMul_1/ReadVariableOp?
5sequential_12/simple_rnn_4/simple_rnn_cell_8/MatMul_1	MLCMatMul)sequential_12/simple_rnn_4/zeros:output:0Lsequential_12/simple_rnn_4/simple_rnn_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????27
5sequential_12/simple_rnn_4/simple_rnn_cell_8/MatMul_1?
0sequential_12/simple_rnn_4/simple_rnn_cell_8/addAddV2=sequential_12/simple_rnn_4/simple_rnn_cell_8/BiasAdd:output:0?sequential_12/simple_rnn_4/simple_rnn_cell_8/MatMul_1:product:0*
T0*(
_output_shapes
:??????????22
0sequential_12/simple_rnn_4/simple_rnn_cell_8/add?
1sequential_12/simple_rnn_4/simple_rnn_cell_8/TanhTanh4sequential_12/simple_rnn_4/simple_rnn_cell_8/add:z:0*
T0*(
_output_shapes
:??????????23
1sequential_12/simple_rnn_4/simple_rnn_cell_8/Tanh?
8sequential_12/simple_rnn_4/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2:
8sequential_12/simple_rnn_4/TensorArrayV2_1/element_shape?
*sequential_12/simple_rnn_4/TensorArrayV2_1TensorListReserveAsequential_12/simple_rnn_4/TensorArrayV2_1/element_shape:output:03sequential_12/simple_rnn_4/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02,
*sequential_12/simple_rnn_4/TensorArrayV2_1?
sequential_12/simple_rnn_4/timeConst*
_output_shapes
: *
dtype0*
value	B : 2!
sequential_12/simple_rnn_4/time?
3sequential_12/simple_rnn_4/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????25
3sequential_12/simple_rnn_4/while/maximum_iterations?
-sequential_12/simple_rnn_4/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_12/simple_rnn_4/while/loop_counter?
 sequential_12/simple_rnn_4/whileWhile6sequential_12/simple_rnn_4/while/loop_counter:output:0<sequential_12/simple_rnn_4/while/maximum_iterations:output:0(sequential_12/simple_rnn_4/time:output:03sequential_12/simple_rnn_4/TensorArrayV2_1:handle:0)sequential_12/simple_rnn_4/zeros:output:03sequential_12/simple_rnn_4/strided_slice_1:output:0Rsequential_12/simple_rnn_4/TensorArrayUnstack/TensorListFromTensor:output_handle:0Ksequential_12_simple_rnn_4_simple_rnn_cell_8_matmul_readvariableop_resourceLsequential_12_simple_rnn_4_simple_rnn_cell_8_biasadd_readvariableop_resourceMsequential_12_simple_rnn_4_simple_rnn_cell_8_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :??????????: : : : : *%
_read_only_resource_inputs
	*7
body/R-
+sequential_12_simple_rnn_4_while_body_98638*7
cond/R-
+sequential_12_simple_rnn_4_while_cond_98637*9
output_shapes(
&: : : : :??????????: : : : : *
parallel_iterations 2"
 sequential_12/simple_rnn_4/while?
Ksequential_12/simple_rnn_4/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2M
Ksequential_12/simple_rnn_4/TensorArrayV2Stack/TensorListStack/element_shape?
=sequential_12/simple_rnn_4/TensorArrayV2Stack/TensorListStackTensorListStack)sequential_12/simple_rnn_4/while:output:3Tsequential_12/simple_rnn_4/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:H??????????*
element_dtype02?
=sequential_12/simple_rnn_4/TensorArrayV2Stack/TensorListStack?
0sequential_12/simple_rnn_4/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????22
0sequential_12/simple_rnn_4/strided_slice_3/stack?
2sequential_12/simple_rnn_4/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 24
2sequential_12/simple_rnn_4/strided_slice_3/stack_1?
2sequential_12/simple_rnn_4/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2sequential_12/simple_rnn_4/strided_slice_3/stack_2?
*sequential_12/simple_rnn_4/strided_slice_3StridedSliceFsequential_12/simple_rnn_4/TensorArrayV2Stack/TensorListStack:tensor:09sequential_12/simple_rnn_4/strided_slice_3/stack:output:0;sequential_12/simple_rnn_4/strided_slice_3/stack_1:output:0;sequential_12/simple_rnn_4/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2,
*sequential_12/simple_rnn_4/strided_slice_3?
+sequential_12/simple_rnn_4/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2-
+sequential_12/simple_rnn_4/transpose_1/perm?
&sequential_12/simple_rnn_4/transpose_1	TransposeFsequential_12/simple_rnn_4/TensorArrayV2Stack/TensorListStack:tensor:04sequential_12/simple_rnn_4/transpose_1/perm:output:0*
T0*,
_output_shapes
:?????????H?2(
&sequential_12/simple_rnn_4/transpose_1?
 sequential_12/activation_10/ReluRelu*sequential_12/simple_rnn_4/transpose_1:y:0*
T0*,
_output_shapes
:?????????H?2"
 sequential_12/activation_10/Relu?
 sequential_12/dropout_4/IdentityIdentity.sequential_12/activation_10/Relu:activations:0*
T0*,
_output_shapes
:?????????H?2"
 sequential_12/dropout_4/Identity?
 sequential_12/simple_rnn_5/ShapeShape)sequential_12/dropout_4/Identity:output:0*
T0*
_output_shapes
:2"
 sequential_12/simple_rnn_5/Shape?
.sequential_12/simple_rnn_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 20
.sequential_12/simple_rnn_5/strided_slice/stack?
0sequential_12/simple_rnn_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0sequential_12/simple_rnn_5/strided_slice/stack_1?
0sequential_12/simple_rnn_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0sequential_12/simple_rnn_5/strided_slice/stack_2?
(sequential_12/simple_rnn_5/strided_sliceStridedSlice)sequential_12/simple_rnn_5/Shape:output:07sequential_12/simple_rnn_5/strided_slice/stack:output:09sequential_12/simple_rnn_5/strided_slice/stack_1:output:09sequential_12/simple_rnn_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2*
(sequential_12/simple_rnn_5/strided_slice?
&sequential_12/simple_rnn_5/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22(
&sequential_12/simple_rnn_5/zeros/mul/y?
$sequential_12/simple_rnn_5/zeros/mulMul1sequential_12/simple_rnn_5/strided_slice:output:0/sequential_12/simple_rnn_5/zeros/mul/y:output:0*
T0*
_output_shapes
: 2&
$sequential_12/simple_rnn_5/zeros/mul?
'sequential_12/simple_rnn_5/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2)
'sequential_12/simple_rnn_5/zeros/Less/y?
%sequential_12/simple_rnn_5/zeros/LessLess(sequential_12/simple_rnn_5/zeros/mul:z:00sequential_12/simple_rnn_5/zeros/Less/y:output:0*
T0*
_output_shapes
: 2'
%sequential_12/simple_rnn_5/zeros/Less?
)sequential_12/simple_rnn_5/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22+
)sequential_12/simple_rnn_5/zeros/packed/1?
'sequential_12/simple_rnn_5/zeros/packedPack1sequential_12/simple_rnn_5/strided_slice:output:02sequential_12/simple_rnn_5/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2)
'sequential_12/simple_rnn_5/zeros/packed?
&sequential_12/simple_rnn_5/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2(
&sequential_12/simple_rnn_5/zeros/Const?
 sequential_12/simple_rnn_5/zerosFill0sequential_12/simple_rnn_5/zeros/packed:output:0/sequential_12/simple_rnn_5/zeros/Const:output:0*
T0*'
_output_shapes
:?????????22"
 sequential_12/simple_rnn_5/zeros?
)sequential_12/simple_rnn_5/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2+
)sequential_12/simple_rnn_5/transpose/perm?
$sequential_12/simple_rnn_5/transpose	Transpose)sequential_12/dropout_4/Identity:output:02sequential_12/simple_rnn_5/transpose/perm:output:0*
T0*,
_output_shapes
:H??????????2&
$sequential_12/simple_rnn_5/transpose?
"sequential_12/simple_rnn_5/Shape_1Shape(sequential_12/simple_rnn_5/transpose:y:0*
T0*
_output_shapes
:2$
"sequential_12/simple_rnn_5/Shape_1?
0sequential_12/simple_rnn_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0sequential_12/simple_rnn_5/strided_slice_1/stack?
2sequential_12/simple_rnn_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2sequential_12/simple_rnn_5/strided_slice_1/stack_1?
2sequential_12/simple_rnn_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2sequential_12/simple_rnn_5/strided_slice_1/stack_2?
*sequential_12/simple_rnn_5/strided_slice_1StridedSlice+sequential_12/simple_rnn_5/Shape_1:output:09sequential_12/simple_rnn_5/strided_slice_1/stack:output:0;sequential_12/simple_rnn_5/strided_slice_1/stack_1:output:0;sequential_12/simple_rnn_5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*sequential_12/simple_rnn_5/strided_slice_1?
6sequential_12/simple_rnn_5/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????28
6sequential_12/simple_rnn_5/TensorArrayV2/element_shape?
(sequential_12/simple_rnn_5/TensorArrayV2TensorListReserve?sequential_12/simple_rnn_5/TensorArrayV2/element_shape:output:03sequential_12/simple_rnn_5/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02*
(sequential_12/simple_rnn_5/TensorArrayV2?
Psequential_12/simple_rnn_5/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2R
Psequential_12/simple_rnn_5/TensorArrayUnstack/TensorListFromTensor/element_shape?
Bsequential_12/simple_rnn_5/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor(sequential_12/simple_rnn_5/transpose:y:0Ysequential_12/simple_rnn_5/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02D
Bsequential_12/simple_rnn_5/TensorArrayUnstack/TensorListFromTensor?
0sequential_12/simple_rnn_5/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0sequential_12/simple_rnn_5/strided_slice_2/stack?
2sequential_12/simple_rnn_5/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2sequential_12/simple_rnn_5/strided_slice_2/stack_1?
2sequential_12/simple_rnn_5/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2sequential_12/simple_rnn_5/strided_slice_2/stack_2?
*sequential_12/simple_rnn_5/strided_slice_2StridedSlice(sequential_12/simple_rnn_5/transpose:y:09sequential_12/simple_rnn_5/strided_slice_2/stack:output:0;sequential_12/simple_rnn_5/strided_slice_2/stack_1:output:0;sequential_12/simple_rnn_5/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2,
*sequential_12/simple_rnn_5/strided_slice_2?
Bsequential_12/simple_rnn_5/simple_rnn_cell_9/MatMul/ReadVariableOpReadVariableOpKsequential_12_simple_rnn_5_simple_rnn_cell_9_matmul_readvariableop_resource*
_output_shapes
:	?2*
dtype02D
Bsequential_12/simple_rnn_5/simple_rnn_cell_9/MatMul/ReadVariableOp?
3sequential_12/simple_rnn_5/simple_rnn_cell_9/MatMul	MLCMatMul3sequential_12/simple_rnn_5/strided_slice_2:output:0Jsequential_12/simple_rnn_5/simple_rnn_cell_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????225
3sequential_12/simple_rnn_5/simple_rnn_cell_9/MatMul?
Csequential_12/simple_rnn_5/simple_rnn_cell_9/BiasAdd/ReadVariableOpReadVariableOpLsequential_12_simple_rnn_5_simple_rnn_cell_9_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02E
Csequential_12/simple_rnn_5/simple_rnn_cell_9/BiasAdd/ReadVariableOp?
4sequential_12/simple_rnn_5/simple_rnn_cell_9/BiasAddBiasAdd=sequential_12/simple_rnn_5/simple_rnn_cell_9/MatMul:product:0Ksequential_12/simple_rnn_5/simple_rnn_cell_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????226
4sequential_12/simple_rnn_5/simple_rnn_cell_9/BiasAdd?
Dsequential_12/simple_rnn_5/simple_rnn_cell_9/MatMul_1/ReadVariableOpReadVariableOpMsequential_12_simple_rnn_5_simple_rnn_cell_9_matmul_1_readvariableop_resource*
_output_shapes

:22*
dtype02F
Dsequential_12/simple_rnn_5/simple_rnn_cell_9/MatMul_1/ReadVariableOp?
5sequential_12/simple_rnn_5/simple_rnn_cell_9/MatMul_1	MLCMatMul)sequential_12/simple_rnn_5/zeros:output:0Lsequential_12/simple_rnn_5/simple_rnn_cell_9/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????227
5sequential_12/simple_rnn_5/simple_rnn_cell_9/MatMul_1?
0sequential_12/simple_rnn_5/simple_rnn_cell_9/addAddV2=sequential_12/simple_rnn_5/simple_rnn_cell_9/BiasAdd:output:0?sequential_12/simple_rnn_5/simple_rnn_cell_9/MatMul_1:product:0*
T0*'
_output_shapes
:?????????222
0sequential_12/simple_rnn_5/simple_rnn_cell_9/add?
1sequential_12/simple_rnn_5/simple_rnn_cell_9/TanhTanh4sequential_12/simple_rnn_5/simple_rnn_cell_9/add:z:0*
T0*'
_output_shapes
:?????????223
1sequential_12/simple_rnn_5/simple_rnn_cell_9/Tanh?
8sequential_12/simple_rnn_5/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   2:
8sequential_12/simple_rnn_5/TensorArrayV2_1/element_shape?
*sequential_12/simple_rnn_5/TensorArrayV2_1TensorListReserveAsequential_12/simple_rnn_5/TensorArrayV2_1/element_shape:output:03sequential_12/simple_rnn_5/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02,
*sequential_12/simple_rnn_5/TensorArrayV2_1?
sequential_12/simple_rnn_5/timeConst*
_output_shapes
: *
dtype0*
value	B : 2!
sequential_12/simple_rnn_5/time?
3sequential_12/simple_rnn_5/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????25
3sequential_12/simple_rnn_5/while/maximum_iterations?
-sequential_12/simple_rnn_5/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_12/simple_rnn_5/while/loop_counter?
 sequential_12/simple_rnn_5/whileWhile6sequential_12/simple_rnn_5/while/loop_counter:output:0<sequential_12/simple_rnn_5/while/maximum_iterations:output:0(sequential_12/simple_rnn_5/time:output:03sequential_12/simple_rnn_5/TensorArrayV2_1:handle:0)sequential_12/simple_rnn_5/zeros:output:03sequential_12/simple_rnn_5/strided_slice_1:output:0Rsequential_12/simple_rnn_5/TensorArrayUnstack/TensorListFromTensor:output_handle:0Ksequential_12_simple_rnn_5_simple_rnn_cell_9_matmul_readvariableop_resourceLsequential_12_simple_rnn_5_simple_rnn_cell_9_biasadd_readvariableop_resourceMsequential_12_simple_rnn_5_simple_rnn_cell_9_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????2: : : : : *%
_read_only_resource_inputs
	*7
body/R-
+sequential_12_simple_rnn_5_while_body_98748*7
cond/R-
+sequential_12_simple_rnn_5_while_cond_98747*8
output_shapes'
%: : : : :?????????2: : : : : *
parallel_iterations 2"
 sequential_12/simple_rnn_5/while?
Ksequential_12/simple_rnn_5/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   2M
Ksequential_12/simple_rnn_5/TensorArrayV2Stack/TensorListStack/element_shape?
=sequential_12/simple_rnn_5/TensorArrayV2Stack/TensorListStackTensorListStack)sequential_12/simple_rnn_5/while:output:3Tsequential_12/simple_rnn_5/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:H?????????2*
element_dtype02?
=sequential_12/simple_rnn_5/TensorArrayV2Stack/TensorListStack?
0sequential_12/simple_rnn_5/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????22
0sequential_12/simple_rnn_5/strided_slice_3/stack?
2sequential_12/simple_rnn_5/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 24
2sequential_12/simple_rnn_5/strided_slice_3/stack_1?
2sequential_12/simple_rnn_5/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2sequential_12/simple_rnn_5/strided_slice_3/stack_2?
*sequential_12/simple_rnn_5/strided_slice_3StridedSliceFsequential_12/simple_rnn_5/TensorArrayV2Stack/TensorListStack:tensor:09sequential_12/simple_rnn_5/strided_slice_3/stack:output:0;sequential_12/simple_rnn_5/strided_slice_3/stack_1:output:0;sequential_12/simple_rnn_5/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????2*
shrink_axis_mask2,
*sequential_12/simple_rnn_5/strided_slice_3?
+sequential_12/simple_rnn_5/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2-
+sequential_12/simple_rnn_5/transpose_1/perm?
&sequential_12/simple_rnn_5/transpose_1	TransposeFsequential_12/simple_rnn_5/TensorArrayV2Stack/TensorListStack:tensor:04sequential_12/simple_rnn_5/transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????H22(
&sequential_12/simple_rnn_5/transpose_1?
 sequential_12/activation_11/ReluRelu3sequential_12/simple_rnn_5/strided_slice_3:output:0*
T0*'
_output_shapes
:?????????22"
 sequential_12/activation_11/Relu?
 sequential_12/dropout_5/IdentityIdentity.sequential_12/activation_11/Relu:activations:0*
T0*'
_output_shapes
:?????????22"
 sequential_12/dropout_5/Identity?
/sequential_12/dense_14/MLCMatMul/ReadVariableOpReadVariableOp8sequential_12_dense_14_mlcmatmul_readvariableop_resource*
_output_shapes

:2$*
dtype021
/sequential_12/dense_14/MLCMatMul/ReadVariableOp?
 sequential_12/dense_14/MLCMatMul	MLCMatMul)sequential_12/dropout_5/Identity:output:07sequential_12/dense_14/MLCMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????$2"
 sequential_12/dense_14/MLCMatMul?
-sequential_12/dense_14/BiasAdd/ReadVariableOpReadVariableOp6sequential_12_dense_14_biasadd_readvariableop_resource*
_output_shapes
:$*
dtype02/
-sequential_12/dense_14/BiasAdd/ReadVariableOp?
sequential_12/dense_14/BiasAddBiasAdd*sequential_12/dense_14/MLCMatMul:product:05sequential_12/dense_14/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????$2 
sequential_12/dense_14/BiasAdd?
sequential_12/dense_14/TanhTanh'sequential_12/dense_14/BiasAdd:output:0*
T0*'
_output_shapes
:?????????$2
sequential_12/dense_14/Tanh?
IdentityIdentitysequential_12/dense_14/Tanh:y:0.^sequential_12/dense_14/BiasAdd/ReadVariableOp0^sequential_12/dense_14/MLCMatMul/ReadVariableOpD^sequential_12/simple_rnn_4/simple_rnn_cell_8/BiasAdd/ReadVariableOpC^sequential_12/simple_rnn_4/simple_rnn_cell_8/MatMul/ReadVariableOpE^sequential_12/simple_rnn_4/simple_rnn_cell_8/MatMul_1/ReadVariableOp!^sequential_12/simple_rnn_4/whileD^sequential_12/simple_rnn_5/simple_rnn_cell_9/BiasAdd/ReadVariableOpC^sequential_12/simple_rnn_5/simple_rnn_cell_9/MatMul/ReadVariableOpE^sequential_12/simple_rnn_5/simple_rnn_cell_9/MatMul_1/ReadVariableOp!^sequential_12/simple_rnn_5/while*
T0*'
_output_shapes
:?????????$2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:?????????H::::::::2^
-sequential_12/dense_14/BiasAdd/ReadVariableOp-sequential_12/dense_14/BiasAdd/ReadVariableOp2b
/sequential_12/dense_14/MLCMatMul/ReadVariableOp/sequential_12/dense_14/MLCMatMul/ReadVariableOp2?
Csequential_12/simple_rnn_4/simple_rnn_cell_8/BiasAdd/ReadVariableOpCsequential_12/simple_rnn_4/simple_rnn_cell_8/BiasAdd/ReadVariableOp2?
Bsequential_12/simple_rnn_4/simple_rnn_cell_8/MatMul/ReadVariableOpBsequential_12/simple_rnn_4/simple_rnn_cell_8/MatMul/ReadVariableOp2?
Dsequential_12/simple_rnn_4/simple_rnn_cell_8/MatMul_1/ReadVariableOpDsequential_12/simple_rnn_4/simple_rnn_cell_8/MatMul_1/ReadVariableOp2D
 sequential_12/simple_rnn_4/while sequential_12/simple_rnn_4/while2?
Csequential_12/simple_rnn_5/simple_rnn_cell_9/BiasAdd/ReadVariableOpCsequential_12/simple_rnn_5/simple_rnn_cell_9/BiasAdd/ReadVariableOp2?
Bsequential_12/simple_rnn_5/simple_rnn_cell_9/MatMul/ReadVariableOpBsequential_12/simple_rnn_5/simple_rnn_cell_9/MatMul/ReadVariableOp2?
Dsequential_12/simple_rnn_5/simple_rnn_cell_9/MatMul_1/ReadVariableOpDsequential_12/simple_rnn_5/simple_rnn_cell_9/MatMul_1/ReadVariableOp2D
 sequential_12/simple_rnn_5/while sequential_12/simple_rnn_5/while:_ [
+
_output_shapes
:?????????H
,
_user_specified_namesimple_rnn_4_input
?<
?
G__inference_simple_rnn_5_layer_call_and_return_conditional_losses_99721

inputs
simple_rnn_cell_9_99646
simple_rnn_cell_9_99648
simple_rnn_cell_9_99650
identity??)simple_rnn_cell_9/StatefulPartitionedCall?whileD
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
)simple_rnn_cell_9/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0simple_rnn_cell_9_99646simple_rnn_cell_9_99648simple_rnn_cell_9_99650*
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
GPU 2J 8? *U
fPRN
L__inference_simple_rnn_cell_9_layer_call_and_return_conditional_losses_993842+
)simple_rnn_cell_9/StatefulPartitionedCall?
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0simple_rnn_cell_9_99646simple_rnn_cell_9_99648simple_rnn_cell_9_99650*
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
bodyR
while_body_99658*
condR
while_cond_99657*8
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
IdentityIdentitystrided_slice_3:output:0*^simple_rnn_cell_9/StatefulPartitionedCall^while*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*@
_input_shapes/
-:???????????????????:::2V
)simple_rnn_cell_9/StatefulPartitionedCall)simple_rnn_cell_9/StatefulPartitionedCall2
whilewhile:] Y
5
_output_shapes#
!:???????????????????
 
_user_specified_nameinputs
?
?
-__inference_simple_rnn_5_layer_call_fn_102150

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
H__inference_simple_rnn_5_layer_call_and_return_conditional_losses_1003672
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????H?:::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:?????????H?
 
_user_specified_nameinputs
?
?
$__inference_signature_wrapper_100620
simple_rnn_4_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallsimple_rnn_4_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????$**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *)
f$R"
 __inference__wrapped_model_988232
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????$2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:?????????H::::::::22
StatefulPartitionedCallStatefulPartitionedCall:_ [
+
_output_shapes
:?????????H
,
_user_specified_namesimple_rnn_4_input
?R
?
+sequential_12_simple_rnn_4_while_body_98638R
Nsequential_12_simple_rnn_4_while_sequential_12_simple_rnn_4_while_loop_counterX
Tsequential_12_simple_rnn_4_while_sequential_12_simple_rnn_4_while_maximum_iterations0
,sequential_12_simple_rnn_4_while_placeholder2
.sequential_12_simple_rnn_4_while_placeholder_12
.sequential_12_simple_rnn_4_while_placeholder_2Q
Msequential_12_simple_rnn_4_while_sequential_12_simple_rnn_4_strided_slice_1_0?
?sequential_12_simple_rnn_4_while_tensorarrayv2read_tensorlistgetitem_sequential_12_simple_rnn_4_tensorarrayunstack_tensorlistfromtensor_0W
Ssequential_12_simple_rnn_4_while_simple_rnn_cell_8_matmul_readvariableop_resource_0X
Tsequential_12_simple_rnn_4_while_simple_rnn_cell_8_biasadd_readvariableop_resource_0Y
Usequential_12_simple_rnn_4_while_simple_rnn_cell_8_matmul_1_readvariableop_resource_0-
)sequential_12_simple_rnn_4_while_identity/
+sequential_12_simple_rnn_4_while_identity_1/
+sequential_12_simple_rnn_4_while_identity_2/
+sequential_12_simple_rnn_4_while_identity_3/
+sequential_12_simple_rnn_4_while_identity_4O
Ksequential_12_simple_rnn_4_while_sequential_12_simple_rnn_4_strided_slice_1?
?sequential_12_simple_rnn_4_while_tensorarrayv2read_tensorlistgetitem_sequential_12_simple_rnn_4_tensorarrayunstack_tensorlistfromtensorU
Qsequential_12_simple_rnn_4_while_simple_rnn_cell_8_matmul_readvariableop_resourceV
Rsequential_12_simple_rnn_4_while_simple_rnn_cell_8_biasadd_readvariableop_resourceW
Ssequential_12_simple_rnn_4_while_simple_rnn_cell_8_matmul_1_readvariableop_resource??Isequential_12/simple_rnn_4/while/simple_rnn_cell_8/BiasAdd/ReadVariableOp?Hsequential_12/simple_rnn_4/while/simple_rnn_cell_8/MatMul/ReadVariableOp?Jsequential_12/simple_rnn_4/while/simple_rnn_cell_8/MatMul_1/ReadVariableOp?
Rsequential_12/simple_rnn_4/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2T
Rsequential_12/simple_rnn_4/while/TensorArrayV2Read/TensorListGetItem/element_shape?
Dsequential_12/simple_rnn_4/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem?sequential_12_simple_rnn_4_while_tensorarrayv2read_tensorlistgetitem_sequential_12_simple_rnn_4_tensorarrayunstack_tensorlistfromtensor_0,sequential_12_simple_rnn_4_while_placeholder[sequential_12/simple_rnn_4/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02F
Dsequential_12/simple_rnn_4/while/TensorArrayV2Read/TensorListGetItem?
Hsequential_12/simple_rnn_4/while/simple_rnn_cell_8/MatMul/ReadVariableOpReadVariableOpSsequential_12_simple_rnn_4_while_simple_rnn_cell_8_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype02J
Hsequential_12/simple_rnn_4/while/simple_rnn_cell_8/MatMul/ReadVariableOp?
9sequential_12/simple_rnn_4/while/simple_rnn_cell_8/MatMul	MLCMatMulKsequential_12/simple_rnn_4/while/TensorArrayV2Read/TensorListGetItem:item:0Psequential_12/simple_rnn_4/while/simple_rnn_cell_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2;
9sequential_12/simple_rnn_4/while/simple_rnn_cell_8/MatMul?
Isequential_12/simple_rnn_4/while/simple_rnn_cell_8/BiasAdd/ReadVariableOpReadVariableOpTsequential_12_simple_rnn_4_while_simple_rnn_cell_8_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02K
Isequential_12/simple_rnn_4/while/simple_rnn_cell_8/BiasAdd/ReadVariableOp?
:sequential_12/simple_rnn_4/while/simple_rnn_cell_8/BiasAddBiasAddCsequential_12/simple_rnn_4/while/simple_rnn_cell_8/MatMul:product:0Qsequential_12/simple_rnn_4/while/simple_rnn_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2<
:sequential_12/simple_rnn_4/while/simple_rnn_cell_8/BiasAdd?
Jsequential_12/simple_rnn_4/while/simple_rnn_cell_8/MatMul_1/ReadVariableOpReadVariableOpUsequential_12_simple_rnn_4_while_simple_rnn_cell_8_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02L
Jsequential_12/simple_rnn_4/while/simple_rnn_cell_8/MatMul_1/ReadVariableOp?
;sequential_12/simple_rnn_4/while/simple_rnn_cell_8/MatMul_1	MLCMatMul.sequential_12_simple_rnn_4_while_placeholder_2Rsequential_12/simple_rnn_4/while/simple_rnn_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2=
;sequential_12/simple_rnn_4/while/simple_rnn_cell_8/MatMul_1?
6sequential_12/simple_rnn_4/while/simple_rnn_cell_8/addAddV2Csequential_12/simple_rnn_4/while/simple_rnn_cell_8/BiasAdd:output:0Esequential_12/simple_rnn_4/while/simple_rnn_cell_8/MatMul_1:product:0*
T0*(
_output_shapes
:??????????28
6sequential_12/simple_rnn_4/while/simple_rnn_cell_8/add?
7sequential_12/simple_rnn_4/while/simple_rnn_cell_8/TanhTanh:sequential_12/simple_rnn_4/while/simple_rnn_cell_8/add:z:0*
T0*(
_output_shapes
:??????????29
7sequential_12/simple_rnn_4/while/simple_rnn_cell_8/Tanh?
Esequential_12/simple_rnn_4/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem.sequential_12_simple_rnn_4_while_placeholder_1,sequential_12_simple_rnn_4_while_placeholder;sequential_12/simple_rnn_4/while/simple_rnn_cell_8/Tanh:y:0*
_output_shapes
: *
element_dtype02G
Esequential_12/simple_rnn_4/while/TensorArrayV2Write/TensorListSetItem?
&sequential_12/simple_rnn_4/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2(
&sequential_12/simple_rnn_4/while/add/y?
$sequential_12/simple_rnn_4/while/addAddV2,sequential_12_simple_rnn_4_while_placeholder/sequential_12/simple_rnn_4/while/add/y:output:0*
T0*
_output_shapes
: 2&
$sequential_12/simple_rnn_4/while/add?
(sequential_12/simple_rnn_4/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2*
(sequential_12/simple_rnn_4/while/add_1/y?
&sequential_12/simple_rnn_4/while/add_1AddV2Nsequential_12_simple_rnn_4_while_sequential_12_simple_rnn_4_while_loop_counter1sequential_12/simple_rnn_4/while/add_1/y:output:0*
T0*
_output_shapes
: 2(
&sequential_12/simple_rnn_4/while/add_1?
)sequential_12/simple_rnn_4/while/IdentityIdentity*sequential_12/simple_rnn_4/while/add_1:z:0J^sequential_12/simple_rnn_4/while/simple_rnn_cell_8/BiasAdd/ReadVariableOpI^sequential_12/simple_rnn_4/while/simple_rnn_cell_8/MatMul/ReadVariableOpK^sequential_12/simple_rnn_4/while/simple_rnn_cell_8/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2+
)sequential_12/simple_rnn_4/while/Identity?
+sequential_12/simple_rnn_4/while/Identity_1IdentityTsequential_12_simple_rnn_4_while_sequential_12_simple_rnn_4_while_maximum_iterationsJ^sequential_12/simple_rnn_4/while/simple_rnn_cell_8/BiasAdd/ReadVariableOpI^sequential_12/simple_rnn_4/while/simple_rnn_cell_8/MatMul/ReadVariableOpK^sequential_12/simple_rnn_4/while/simple_rnn_cell_8/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2-
+sequential_12/simple_rnn_4/while/Identity_1?
+sequential_12/simple_rnn_4/while/Identity_2Identity(sequential_12/simple_rnn_4/while/add:z:0J^sequential_12/simple_rnn_4/while/simple_rnn_cell_8/BiasAdd/ReadVariableOpI^sequential_12/simple_rnn_4/while/simple_rnn_cell_8/MatMul/ReadVariableOpK^sequential_12/simple_rnn_4/while/simple_rnn_cell_8/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2-
+sequential_12/simple_rnn_4/while/Identity_2?
+sequential_12/simple_rnn_4/while/Identity_3IdentityUsequential_12/simple_rnn_4/while/TensorArrayV2Write/TensorListSetItem:output_handle:0J^sequential_12/simple_rnn_4/while/simple_rnn_cell_8/BiasAdd/ReadVariableOpI^sequential_12/simple_rnn_4/while/simple_rnn_cell_8/MatMul/ReadVariableOpK^sequential_12/simple_rnn_4/while/simple_rnn_cell_8/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2-
+sequential_12/simple_rnn_4/while/Identity_3?
+sequential_12/simple_rnn_4/while/Identity_4Identity;sequential_12/simple_rnn_4/while/simple_rnn_cell_8/Tanh:y:0J^sequential_12/simple_rnn_4/while/simple_rnn_cell_8/BiasAdd/ReadVariableOpI^sequential_12/simple_rnn_4/while/simple_rnn_cell_8/MatMul/ReadVariableOpK^sequential_12/simple_rnn_4/while/simple_rnn_cell_8/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2-
+sequential_12/simple_rnn_4/while/Identity_4"_
)sequential_12_simple_rnn_4_while_identity2sequential_12/simple_rnn_4/while/Identity:output:0"c
+sequential_12_simple_rnn_4_while_identity_14sequential_12/simple_rnn_4/while/Identity_1:output:0"c
+sequential_12_simple_rnn_4_while_identity_24sequential_12/simple_rnn_4/while/Identity_2:output:0"c
+sequential_12_simple_rnn_4_while_identity_34sequential_12/simple_rnn_4/while/Identity_3:output:0"c
+sequential_12_simple_rnn_4_while_identity_44sequential_12/simple_rnn_4/while/Identity_4:output:0"?
Ksequential_12_simple_rnn_4_while_sequential_12_simple_rnn_4_strided_slice_1Msequential_12_simple_rnn_4_while_sequential_12_simple_rnn_4_strided_slice_1_0"?
Rsequential_12_simple_rnn_4_while_simple_rnn_cell_8_biasadd_readvariableop_resourceTsequential_12_simple_rnn_4_while_simple_rnn_cell_8_biasadd_readvariableop_resource_0"?
Ssequential_12_simple_rnn_4_while_simple_rnn_cell_8_matmul_1_readvariableop_resourceUsequential_12_simple_rnn_4_while_simple_rnn_cell_8_matmul_1_readvariableop_resource_0"?
Qsequential_12_simple_rnn_4_while_simple_rnn_cell_8_matmul_readvariableop_resourceSsequential_12_simple_rnn_4_while_simple_rnn_cell_8_matmul_readvariableop_resource_0"?
?sequential_12_simple_rnn_4_while_tensorarrayv2read_tensorlistgetitem_sequential_12_simple_rnn_4_tensorarrayunstack_tensorlistfromtensor?sequential_12_simple_rnn_4_while_tensorarrayv2read_tensorlistgetitem_sequential_12_simple_rnn_4_tensorarrayunstack_tensorlistfromtensor_0*?
_input_shapes.
,: : : : :??????????: : :::2?
Isequential_12/simple_rnn_4/while/simple_rnn_cell_8/BiasAdd/ReadVariableOpIsequential_12/simple_rnn_4/while/simple_rnn_cell_8/BiasAdd/ReadVariableOp2?
Hsequential_12/simple_rnn_4/while/simple_rnn_cell_8/MatMul/ReadVariableOpHsequential_12/simple_rnn_4/while/simple_rnn_cell_8/MatMul/ReadVariableOp2?
Jsequential_12/simple_rnn_4/while/simple_rnn_cell_8/MatMul_1/ReadVariableOpJsequential_12/simple_rnn_4/while/simple_rnn_cell_8/MatMul_1/ReadVariableOp: 
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
E__inference_dropout_5_layer_call_and_return_conditional_losses_100424

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
?B
?
simple_rnn_4_while_body_1006666
2simple_rnn_4_while_simple_rnn_4_while_loop_counter<
8simple_rnn_4_while_simple_rnn_4_while_maximum_iterations"
simple_rnn_4_while_placeholder$
 simple_rnn_4_while_placeholder_1$
 simple_rnn_4_while_placeholder_25
1simple_rnn_4_while_simple_rnn_4_strided_slice_1_0q
msimple_rnn_4_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_4_tensorarrayunstack_tensorlistfromtensor_0I
Esimple_rnn_4_while_simple_rnn_cell_8_matmul_readvariableop_resource_0J
Fsimple_rnn_4_while_simple_rnn_cell_8_biasadd_readvariableop_resource_0K
Gsimple_rnn_4_while_simple_rnn_cell_8_matmul_1_readvariableop_resource_0
simple_rnn_4_while_identity!
simple_rnn_4_while_identity_1!
simple_rnn_4_while_identity_2!
simple_rnn_4_while_identity_3!
simple_rnn_4_while_identity_43
/simple_rnn_4_while_simple_rnn_4_strided_slice_1o
ksimple_rnn_4_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_4_tensorarrayunstack_tensorlistfromtensorG
Csimple_rnn_4_while_simple_rnn_cell_8_matmul_readvariableop_resourceH
Dsimple_rnn_4_while_simple_rnn_cell_8_biasadd_readvariableop_resourceI
Esimple_rnn_4_while_simple_rnn_cell_8_matmul_1_readvariableop_resource??;simple_rnn_4/while/simple_rnn_cell_8/BiasAdd/ReadVariableOp?:simple_rnn_4/while/simple_rnn_cell_8/MatMul/ReadVariableOp?<simple_rnn_4/while/simple_rnn_cell_8/MatMul_1/ReadVariableOp?
Dsimple_rnn_4/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2F
Dsimple_rnn_4/while/TensorArrayV2Read/TensorListGetItem/element_shape?
6simple_rnn_4/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemmsimple_rnn_4_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_4_tensorarrayunstack_tensorlistfromtensor_0simple_rnn_4_while_placeholderMsimple_rnn_4/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype028
6simple_rnn_4/while/TensorArrayV2Read/TensorListGetItem?
:simple_rnn_4/while/simple_rnn_cell_8/MatMul/ReadVariableOpReadVariableOpEsimple_rnn_4_while_simple_rnn_cell_8_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype02<
:simple_rnn_4/while/simple_rnn_cell_8/MatMul/ReadVariableOp?
+simple_rnn_4/while/simple_rnn_cell_8/MatMul	MLCMatMul=simple_rnn_4/while/TensorArrayV2Read/TensorListGetItem:item:0Bsimple_rnn_4/while/simple_rnn_cell_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2-
+simple_rnn_4/while/simple_rnn_cell_8/MatMul?
;simple_rnn_4/while/simple_rnn_cell_8/BiasAdd/ReadVariableOpReadVariableOpFsimple_rnn_4_while_simple_rnn_cell_8_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02=
;simple_rnn_4/while/simple_rnn_cell_8/BiasAdd/ReadVariableOp?
,simple_rnn_4/while/simple_rnn_cell_8/BiasAddBiasAdd5simple_rnn_4/while/simple_rnn_cell_8/MatMul:product:0Csimple_rnn_4/while/simple_rnn_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2.
,simple_rnn_4/while/simple_rnn_cell_8/BiasAdd?
<simple_rnn_4/while/simple_rnn_cell_8/MatMul_1/ReadVariableOpReadVariableOpGsimple_rnn_4_while_simple_rnn_cell_8_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02>
<simple_rnn_4/while/simple_rnn_cell_8/MatMul_1/ReadVariableOp?
-simple_rnn_4/while/simple_rnn_cell_8/MatMul_1	MLCMatMul simple_rnn_4_while_placeholder_2Dsimple_rnn_4/while/simple_rnn_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2/
-simple_rnn_4/while/simple_rnn_cell_8/MatMul_1?
(simple_rnn_4/while/simple_rnn_cell_8/addAddV25simple_rnn_4/while/simple_rnn_cell_8/BiasAdd:output:07simple_rnn_4/while/simple_rnn_cell_8/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2*
(simple_rnn_4/while/simple_rnn_cell_8/add?
)simple_rnn_4/while/simple_rnn_cell_8/TanhTanh,simple_rnn_4/while/simple_rnn_cell_8/add:z:0*
T0*(
_output_shapes
:??????????2+
)simple_rnn_4/while/simple_rnn_cell_8/Tanh?
7simple_rnn_4/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem simple_rnn_4_while_placeholder_1simple_rnn_4_while_placeholder-simple_rnn_4/while/simple_rnn_cell_8/Tanh:y:0*
_output_shapes
: *
element_dtype029
7simple_rnn_4/while/TensorArrayV2Write/TensorListSetItemv
simple_rnn_4/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
simple_rnn_4/while/add/y?
simple_rnn_4/while/addAddV2simple_rnn_4_while_placeholder!simple_rnn_4/while/add/y:output:0*
T0*
_output_shapes
: 2
simple_rnn_4/while/addz
simple_rnn_4/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
simple_rnn_4/while/add_1/y?
simple_rnn_4/while/add_1AddV22simple_rnn_4_while_simple_rnn_4_while_loop_counter#simple_rnn_4/while/add_1/y:output:0*
T0*
_output_shapes
: 2
simple_rnn_4/while/add_1?
simple_rnn_4/while/IdentityIdentitysimple_rnn_4/while/add_1:z:0<^simple_rnn_4/while/simple_rnn_cell_8/BiasAdd/ReadVariableOp;^simple_rnn_4/while/simple_rnn_cell_8/MatMul/ReadVariableOp=^simple_rnn_4/while/simple_rnn_cell_8/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
simple_rnn_4/while/Identity?
simple_rnn_4/while/Identity_1Identity8simple_rnn_4_while_simple_rnn_4_while_maximum_iterations<^simple_rnn_4/while/simple_rnn_cell_8/BiasAdd/ReadVariableOp;^simple_rnn_4/while/simple_rnn_cell_8/MatMul/ReadVariableOp=^simple_rnn_4/while/simple_rnn_cell_8/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
simple_rnn_4/while/Identity_1?
simple_rnn_4/while/Identity_2Identitysimple_rnn_4/while/add:z:0<^simple_rnn_4/while/simple_rnn_cell_8/BiasAdd/ReadVariableOp;^simple_rnn_4/while/simple_rnn_cell_8/MatMul/ReadVariableOp=^simple_rnn_4/while/simple_rnn_cell_8/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
simple_rnn_4/while/Identity_2?
simple_rnn_4/while/Identity_3IdentityGsimple_rnn_4/while/TensorArrayV2Write/TensorListSetItem:output_handle:0<^simple_rnn_4/while/simple_rnn_cell_8/BiasAdd/ReadVariableOp;^simple_rnn_4/while/simple_rnn_cell_8/MatMul/ReadVariableOp=^simple_rnn_4/while/simple_rnn_cell_8/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
simple_rnn_4/while/Identity_3?
simple_rnn_4/while/Identity_4Identity-simple_rnn_4/while/simple_rnn_cell_8/Tanh:y:0<^simple_rnn_4/while/simple_rnn_cell_8/BiasAdd/ReadVariableOp;^simple_rnn_4/while/simple_rnn_cell_8/MatMul/ReadVariableOp=^simple_rnn_4/while/simple_rnn_cell_8/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2
simple_rnn_4/while/Identity_4"C
simple_rnn_4_while_identity$simple_rnn_4/while/Identity:output:0"G
simple_rnn_4_while_identity_1&simple_rnn_4/while/Identity_1:output:0"G
simple_rnn_4_while_identity_2&simple_rnn_4/while/Identity_2:output:0"G
simple_rnn_4_while_identity_3&simple_rnn_4/while/Identity_3:output:0"G
simple_rnn_4_while_identity_4&simple_rnn_4/while/Identity_4:output:0"d
/simple_rnn_4_while_simple_rnn_4_strided_slice_11simple_rnn_4_while_simple_rnn_4_strided_slice_1_0"?
Dsimple_rnn_4_while_simple_rnn_cell_8_biasadd_readvariableop_resourceFsimple_rnn_4_while_simple_rnn_cell_8_biasadd_readvariableop_resource_0"?
Esimple_rnn_4_while_simple_rnn_cell_8_matmul_1_readvariableop_resourceGsimple_rnn_4_while_simple_rnn_cell_8_matmul_1_readvariableop_resource_0"?
Csimple_rnn_4_while_simple_rnn_cell_8_matmul_readvariableop_resourceEsimple_rnn_4_while_simple_rnn_cell_8_matmul_readvariableop_resource_0"?
ksimple_rnn_4_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_4_tensorarrayunstack_tensorlistfromtensormsimple_rnn_4_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_4_tensorarrayunstack_tensorlistfromtensor_0*?
_input_shapes.
,: : : : :??????????: : :::2z
;simple_rnn_4/while/simple_rnn_cell_8/BiasAdd/ReadVariableOp;simple_rnn_4/while/simple_rnn_cell_8/BiasAdd/ReadVariableOp2x
:simple_rnn_4/while/simple_rnn_cell_8/MatMul/ReadVariableOp:simple_rnn_4/while/simple_rnn_cell_8/MatMul/ReadVariableOp2|
<simple_rnn_4/while/simple_rnn_cell_8/MatMul_1/ReadVariableOp<simple_rnn_4/while/simple_rnn_cell_8/MatMul_1/ReadVariableOp: 
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
?
+sequential_12_simple_rnn_5_while_cond_98747R
Nsequential_12_simple_rnn_5_while_sequential_12_simple_rnn_5_while_loop_counterX
Tsequential_12_simple_rnn_5_while_sequential_12_simple_rnn_5_while_maximum_iterations0
,sequential_12_simple_rnn_5_while_placeholder2
.sequential_12_simple_rnn_5_while_placeholder_12
.sequential_12_simple_rnn_5_while_placeholder_2T
Psequential_12_simple_rnn_5_while_less_sequential_12_simple_rnn_5_strided_slice_1i
esequential_12_simple_rnn_5_while_sequential_12_simple_rnn_5_while_cond_98747___redundant_placeholder0i
esequential_12_simple_rnn_5_while_sequential_12_simple_rnn_5_while_cond_98747___redundant_placeholder1i
esequential_12_simple_rnn_5_while_sequential_12_simple_rnn_5_while_cond_98747___redundant_placeholder2i
esequential_12_simple_rnn_5_while_sequential_12_simple_rnn_5_while_cond_98747___redundant_placeholder3-
)sequential_12_simple_rnn_5_while_identity
?
%sequential_12/simple_rnn_5/while/LessLess,sequential_12_simple_rnn_5_while_placeholderPsequential_12_simple_rnn_5_while_less_sequential_12_simple_rnn_5_strided_slice_1*
T0*
_output_shapes
: 2'
%sequential_12/simple_rnn_5/while/Less?
)sequential_12/simple_rnn_5/while/IdentityIdentity)sequential_12/simple_rnn_5/while/Less:z:0*
T0
*
_output_shapes
: 2+
)sequential_12/simple_rnn_5/while/Identity"_
)sequential_12_simple_rnn_5_while_identity2sequential_12/simple_rnn_5/while/Identity:output:0*@
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
?
I__inference_sequential_12_layer_call_and_return_conditional_losses_100522

inputs
simple_rnn_4_100498
simple_rnn_4_100500
simple_rnn_4_100502
simple_rnn_5_100507
simple_rnn_5_100509
simple_rnn_5_100511
dense_14_100516
dense_14_100518
identity?? dense_14/StatefulPartitionedCall?$simple_rnn_4/StatefulPartitionedCall?$simple_rnn_5/StatefulPartitionedCall?
$simple_rnn_4/StatefulPartitionedCallStatefulPartitionedCallinputssimple_rnn_4_100498simple_rnn_4_100500simple_rnn_4_100502*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????H?*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_simple_rnn_4_layer_call_and_return_conditional_losses_999622&
$simple_rnn_4/StatefulPartitionedCall?
activation_10/PartitionedCallPartitionedCall-simple_rnn_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????H?* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_activation_10_layer_call_and_return_conditional_losses_1001092
activation_10/PartitionedCall?
dropout_4/PartitionedCallPartitionedCall&activation_10/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????H?* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_4_layer_call_and_return_conditional_losses_1001262
dropout_4/PartitionedCall?
$simple_rnn_5/StatefulPartitionedCallStatefulPartitionedCall"dropout_4/PartitionedCall:output:0simple_rnn_5_100507simple_rnn_5_100509simple_rnn_5_100511*
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
H__inference_simple_rnn_5_layer_call_and_return_conditional_losses_1002552&
$simple_rnn_5/StatefulPartitionedCall?
activation_11/PartitionedCallPartitionedCall-simple_rnn_5/StatefulPartitionedCall:output:0*
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
I__inference_activation_11_layer_call_and_return_conditional_losses_1004022
activation_11/PartitionedCall?
dropout_5/PartitionedCallPartitionedCall&activation_11/PartitionedCall:output:0*
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
E__inference_dropout_5_layer_call_and_return_conditional_losses_1004192
dropout_5/PartitionedCall?
 dense_14/StatefulPartitionedCallStatefulPartitionedCall"dropout_5/PartitionedCall:output:0dense_14_100516dense_14_100518*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????$*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_14_layer_call_and_return_conditional_losses_1004482"
 dense_14/StatefulPartitionedCall?
IdentityIdentity)dense_14/StatefulPartitionedCall:output:0!^dense_14/StatefulPartitionedCall%^simple_rnn_4/StatefulPartitionedCall%^simple_rnn_5/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????$2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:?????????H::::::::2D
 dense_14/StatefulPartitionedCall dense_14/StatefulPartitionedCall2L
$simple_rnn_4/StatefulPartitionedCall$simple_rnn_4/StatefulPartitionedCall2L
$simple_rnn_5/StatefulPartitionedCall$simple_rnn_5/StatefulPartitionedCall:S O
+
_output_shapes
:?????????H
 
_user_specified_nameinputs
?
?
L__inference_simple_rnn_cell_8_layer_call_and_return_conditional_losses_98889

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
?G
?
G__inference_simple_rnn_4_layer_call_and_return_conditional_losses_99962

inputs4
0simple_rnn_cell_8_matmul_readvariableop_resource5
1simple_rnn_cell_8_biasadd_readvariableop_resource6
2simple_rnn_cell_8_matmul_1_readvariableop_resource
identity??(simple_rnn_cell_8/BiasAdd/ReadVariableOp?'simple_rnn_cell_8/MatMul/ReadVariableOp?)simple_rnn_cell_8/MatMul_1/ReadVariableOp?whileD
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
:H?????????2
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
'simple_rnn_cell_8/MatMul/ReadVariableOpReadVariableOp0simple_rnn_cell_8_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02)
'simple_rnn_cell_8/MatMul/ReadVariableOp?
simple_rnn_cell_8/MatMul	MLCMatMulstrided_slice_2:output:0/simple_rnn_cell_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
simple_rnn_cell_8/MatMul?
(simple_rnn_cell_8/BiasAdd/ReadVariableOpReadVariableOp1simple_rnn_cell_8_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02*
(simple_rnn_cell_8/BiasAdd/ReadVariableOp?
simple_rnn_cell_8/BiasAddBiasAdd"simple_rnn_cell_8/MatMul:product:00simple_rnn_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
simple_rnn_cell_8/BiasAdd?
)simple_rnn_cell_8/MatMul_1/ReadVariableOpReadVariableOp2simple_rnn_cell_8_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02+
)simple_rnn_cell_8/MatMul_1/ReadVariableOp?
simple_rnn_cell_8/MatMul_1	MLCMatMulzeros:output:01simple_rnn_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
simple_rnn_cell_8/MatMul_1?
simple_rnn_cell_8/addAddV2"simple_rnn_cell_8/BiasAdd:output:0$simple_rnn_cell_8/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
simple_rnn_cell_8/add?
simple_rnn_cell_8/TanhTanhsimple_rnn_cell_8/add:z:0*
T0*(
_output_shapes
:??????????2
simple_rnn_cell_8/Tanh?
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:00simple_rnn_cell_8_matmul_readvariableop_resource1simple_rnn_cell_8_biasadd_readvariableop_resource2simple_rnn_cell_8_matmul_1_readvariableop_resource*
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
bodyR
while_body_99896*
condR
while_cond_99895*9
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
:H??????????*
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
:?????????H?2
transpose_1?
IdentityIdentitytranspose_1:y:0)^simple_rnn_cell_8/BiasAdd/ReadVariableOp(^simple_rnn_cell_8/MatMul/ReadVariableOp*^simple_rnn_cell_8/MatMul_1/ReadVariableOp^while*
T0*,
_output_shapes
:?????????H?2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????H:::2T
(simple_rnn_cell_8/BiasAdd/ReadVariableOp(simple_rnn_cell_8/BiasAdd/ReadVariableOp2R
'simple_rnn_cell_8/MatMul/ReadVariableOp'simple_rnn_cell_8/MatMul/ReadVariableOp2V
)simple_rnn_cell_8/MatMul_1/ReadVariableOp)simple_rnn_cell_8/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:?????????H
 
_user_specified_nameinputs
?
?
while_cond_100007
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_100007___redundant_placeholder04
0while_while_cond_100007___redundant_placeholder14
0while_while_cond_100007___redundant_placeholder24
0while_while_cond_100007___redundant_placeholder3
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
simple_rnn_4_while_cond_1006656
2simple_rnn_4_while_simple_rnn_4_while_loop_counter<
8simple_rnn_4_while_simple_rnn_4_while_maximum_iterations"
simple_rnn_4_while_placeholder$
 simple_rnn_4_while_placeholder_1$
 simple_rnn_4_while_placeholder_28
4simple_rnn_4_while_less_simple_rnn_4_strided_slice_1N
Jsimple_rnn_4_while_simple_rnn_4_while_cond_100665___redundant_placeholder0N
Jsimple_rnn_4_while_simple_rnn_4_while_cond_100665___redundant_placeholder1N
Jsimple_rnn_4_while_simple_rnn_4_while_cond_100665___redundant_placeholder2N
Jsimple_rnn_4_while_simple_rnn_4_while_cond_100665___redundant_placeholder3
simple_rnn_4_while_identity
?
simple_rnn_4/while/LessLesssimple_rnn_4_while_placeholder4simple_rnn_4_while_less_simple_rnn_4_strided_slice_1*
T0*
_output_shapes
: 2
simple_rnn_4/while/Less?
simple_rnn_4/while/IdentityIdentitysimple_rnn_4/while/Less:z:0*
T0
*
_output_shapes
: 2
simple_rnn_4/while/Identity"C
simple_rnn_4_while_identity$simple_rnn_4/while/Identity:output:0*A
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
a
E__inference_dropout_4_layer_call_and_return_conditional_losses_100126

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
:?????????H?2
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
:?????????H?2	
dropouti
IdentityIdentitydropout:output:0*
T0*,
_output_shapes
:?????????H?2

Identity"
identityIdentity:output:0*+
_input_shapes
:?????????H?:T P
,
_output_shapes
:?????????H?
 
_user_specified_nameinputs
?R
?
+sequential_12_simple_rnn_5_while_body_98748R
Nsequential_12_simple_rnn_5_while_sequential_12_simple_rnn_5_while_loop_counterX
Tsequential_12_simple_rnn_5_while_sequential_12_simple_rnn_5_while_maximum_iterations0
,sequential_12_simple_rnn_5_while_placeholder2
.sequential_12_simple_rnn_5_while_placeholder_12
.sequential_12_simple_rnn_5_while_placeholder_2Q
Msequential_12_simple_rnn_5_while_sequential_12_simple_rnn_5_strided_slice_1_0?
?sequential_12_simple_rnn_5_while_tensorarrayv2read_tensorlistgetitem_sequential_12_simple_rnn_5_tensorarrayunstack_tensorlistfromtensor_0W
Ssequential_12_simple_rnn_5_while_simple_rnn_cell_9_matmul_readvariableop_resource_0X
Tsequential_12_simple_rnn_5_while_simple_rnn_cell_9_biasadd_readvariableop_resource_0Y
Usequential_12_simple_rnn_5_while_simple_rnn_cell_9_matmul_1_readvariableop_resource_0-
)sequential_12_simple_rnn_5_while_identity/
+sequential_12_simple_rnn_5_while_identity_1/
+sequential_12_simple_rnn_5_while_identity_2/
+sequential_12_simple_rnn_5_while_identity_3/
+sequential_12_simple_rnn_5_while_identity_4O
Ksequential_12_simple_rnn_5_while_sequential_12_simple_rnn_5_strided_slice_1?
?sequential_12_simple_rnn_5_while_tensorarrayv2read_tensorlistgetitem_sequential_12_simple_rnn_5_tensorarrayunstack_tensorlistfromtensorU
Qsequential_12_simple_rnn_5_while_simple_rnn_cell_9_matmul_readvariableop_resourceV
Rsequential_12_simple_rnn_5_while_simple_rnn_cell_9_biasadd_readvariableop_resourceW
Ssequential_12_simple_rnn_5_while_simple_rnn_cell_9_matmul_1_readvariableop_resource??Isequential_12/simple_rnn_5/while/simple_rnn_cell_9/BiasAdd/ReadVariableOp?Hsequential_12/simple_rnn_5/while/simple_rnn_cell_9/MatMul/ReadVariableOp?Jsequential_12/simple_rnn_5/while/simple_rnn_cell_9/MatMul_1/ReadVariableOp?
Rsequential_12/simple_rnn_5/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2T
Rsequential_12/simple_rnn_5/while/TensorArrayV2Read/TensorListGetItem/element_shape?
Dsequential_12/simple_rnn_5/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem?sequential_12_simple_rnn_5_while_tensorarrayv2read_tensorlistgetitem_sequential_12_simple_rnn_5_tensorarrayunstack_tensorlistfromtensor_0,sequential_12_simple_rnn_5_while_placeholder[sequential_12/simple_rnn_5/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype02F
Dsequential_12/simple_rnn_5/while/TensorArrayV2Read/TensorListGetItem?
Hsequential_12/simple_rnn_5/while/simple_rnn_cell_9/MatMul/ReadVariableOpReadVariableOpSsequential_12_simple_rnn_5_while_simple_rnn_cell_9_matmul_readvariableop_resource_0*
_output_shapes
:	?2*
dtype02J
Hsequential_12/simple_rnn_5/while/simple_rnn_cell_9/MatMul/ReadVariableOp?
9sequential_12/simple_rnn_5/while/simple_rnn_cell_9/MatMul	MLCMatMulKsequential_12/simple_rnn_5/while/TensorArrayV2Read/TensorListGetItem:item:0Psequential_12/simple_rnn_5/while/simple_rnn_cell_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22;
9sequential_12/simple_rnn_5/while/simple_rnn_cell_9/MatMul?
Isequential_12/simple_rnn_5/while/simple_rnn_cell_9/BiasAdd/ReadVariableOpReadVariableOpTsequential_12_simple_rnn_5_while_simple_rnn_cell_9_biasadd_readvariableop_resource_0*
_output_shapes
:2*
dtype02K
Isequential_12/simple_rnn_5/while/simple_rnn_cell_9/BiasAdd/ReadVariableOp?
:sequential_12/simple_rnn_5/while/simple_rnn_cell_9/BiasAddBiasAddCsequential_12/simple_rnn_5/while/simple_rnn_cell_9/MatMul:product:0Qsequential_12/simple_rnn_5/while/simple_rnn_cell_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22<
:sequential_12/simple_rnn_5/while/simple_rnn_cell_9/BiasAdd?
Jsequential_12/simple_rnn_5/while/simple_rnn_cell_9/MatMul_1/ReadVariableOpReadVariableOpUsequential_12_simple_rnn_5_while_simple_rnn_cell_9_matmul_1_readvariableop_resource_0*
_output_shapes

:22*
dtype02L
Jsequential_12/simple_rnn_5/while/simple_rnn_cell_9/MatMul_1/ReadVariableOp?
;sequential_12/simple_rnn_5/while/simple_rnn_cell_9/MatMul_1	MLCMatMul.sequential_12_simple_rnn_5_while_placeholder_2Rsequential_12/simple_rnn_5/while/simple_rnn_cell_9/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22=
;sequential_12/simple_rnn_5/while/simple_rnn_cell_9/MatMul_1?
6sequential_12/simple_rnn_5/while/simple_rnn_cell_9/addAddV2Csequential_12/simple_rnn_5/while/simple_rnn_cell_9/BiasAdd:output:0Esequential_12/simple_rnn_5/while/simple_rnn_cell_9/MatMul_1:product:0*
T0*'
_output_shapes
:?????????228
6sequential_12/simple_rnn_5/while/simple_rnn_cell_9/add?
7sequential_12/simple_rnn_5/while/simple_rnn_cell_9/TanhTanh:sequential_12/simple_rnn_5/while/simple_rnn_cell_9/add:z:0*
T0*'
_output_shapes
:?????????229
7sequential_12/simple_rnn_5/while/simple_rnn_cell_9/Tanh?
Esequential_12/simple_rnn_5/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem.sequential_12_simple_rnn_5_while_placeholder_1,sequential_12_simple_rnn_5_while_placeholder;sequential_12/simple_rnn_5/while/simple_rnn_cell_9/Tanh:y:0*
_output_shapes
: *
element_dtype02G
Esequential_12/simple_rnn_5/while/TensorArrayV2Write/TensorListSetItem?
&sequential_12/simple_rnn_5/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2(
&sequential_12/simple_rnn_5/while/add/y?
$sequential_12/simple_rnn_5/while/addAddV2,sequential_12_simple_rnn_5_while_placeholder/sequential_12/simple_rnn_5/while/add/y:output:0*
T0*
_output_shapes
: 2&
$sequential_12/simple_rnn_5/while/add?
(sequential_12/simple_rnn_5/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2*
(sequential_12/simple_rnn_5/while/add_1/y?
&sequential_12/simple_rnn_5/while/add_1AddV2Nsequential_12_simple_rnn_5_while_sequential_12_simple_rnn_5_while_loop_counter1sequential_12/simple_rnn_5/while/add_1/y:output:0*
T0*
_output_shapes
: 2(
&sequential_12/simple_rnn_5/while/add_1?
)sequential_12/simple_rnn_5/while/IdentityIdentity*sequential_12/simple_rnn_5/while/add_1:z:0J^sequential_12/simple_rnn_5/while/simple_rnn_cell_9/BiasAdd/ReadVariableOpI^sequential_12/simple_rnn_5/while/simple_rnn_cell_9/MatMul/ReadVariableOpK^sequential_12/simple_rnn_5/while/simple_rnn_cell_9/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2+
)sequential_12/simple_rnn_5/while/Identity?
+sequential_12/simple_rnn_5/while/Identity_1IdentityTsequential_12_simple_rnn_5_while_sequential_12_simple_rnn_5_while_maximum_iterationsJ^sequential_12/simple_rnn_5/while/simple_rnn_cell_9/BiasAdd/ReadVariableOpI^sequential_12/simple_rnn_5/while/simple_rnn_cell_9/MatMul/ReadVariableOpK^sequential_12/simple_rnn_5/while/simple_rnn_cell_9/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2-
+sequential_12/simple_rnn_5/while/Identity_1?
+sequential_12/simple_rnn_5/while/Identity_2Identity(sequential_12/simple_rnn_5/while/add:z:0J^sequential_12/simple_rnn_5/while/simple_rnn_cell_9/BiasAdd/ReadVariableOpI^sequential_12/simple_rnn_5/while/simple_rnn_cell_9/MatMul/ReadVariableOpK^sequential_12/simple_rnn_5/while/simple_rnn_cell_9/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2-
+sequential_12/simple_rnn_5/while/Identity_2?
+sequential_12/simple_rnn_5/while/Identity_3IdentityUsequential_12/simple_rnn_5/while/TensorArrayV2Write/TensorListSetItem:output_handle:0J^sequential_12/simple_rnn_5/while/simple_rnn_cell_9/BiasAdd/ReadVariableOpI^sequential_12/simple_rnn_5/while/simple_rnn_cell_9/MatMul/ReadVariableOpK^sequential_12/simple_rnn_5/while/simple_rnn_cell_9/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2-
+sequential_12/simple_rnn_5/while/Identity_3?
+sequential_12/simple_rnn_5/while/Identity_4Identity;sequential_12/simple_rnn_5/while/simple_rnn_cell_9/Tanh:y:0J^sequential_12/simple_rnn_5/while/simple_rnn_cell_9/BiasAdd/ReadVariableOpI^sequential_12/simple_rnn_5/while/simple_rnn_cell_9/MatMul/ReadVariableOpK^sequential_12/simple_rnn_5/while/simple_rnn_cell_9/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????22-
+sequential_12/simple_rnn_5/while/Identity_4"_
)sequential_12_simple_rnn_5_while_identity2sequential_12/simple_rnn_5/while/Identity:output:0"c
+sequential_12_simple_rnn_5_while_identity_14sequential_12/simple_rnn_5/while/Identity_1:output:0"c
+sequential_12_simple_rnn_5_while_identity_24sequential_12/simple_rnn_5/while/Identity_2:output:0"c
+sequential_12_simple_rnn_5_while_identity_34sequential_12/simple_rnn_5/while/Identity_3:output:0"c
+sequential_12_simple_rnn_5_while_identity_44sequential_12/simple_rnn_5/while/Identity_4:output:0"?
Ksequential_12_simple_rnn_5_while_sequential_12_simple_rnn_5_strided_slice_1Msequential_12_simple_rnn_5_while_sequential_12_simple_rnn_5_strided_slice_1_0"?
Rsequential_12_simple_rnn_5_while_simple_rnn_cell_9_biasadd_readvariableop_resourceTsequential_12_simple_rnn_5_while_simple_rnn_cell_9_biasadd_readvariableop_resource_0"?
Ssequential_12_simple_rnn_5_while_simple_rnn_cell_9_matmul_1_readvariableop_resourceUsequential_12_simple_rnn_5_while_simple_rnn_cell_9_matmul_1_readvariableop_resource_0"?
Qsequential_12_simple_rnn_5_while_simple_rnn_cell_9_matmul_readvariableop_resourceSsequential_12_simple_rnn_5_while_simple_rnn_cell_9_matmul_readvariableop_resource_0"?
?sequential_12_simple_rnn_5_while_tensorarrayv2read_tensorlistgetitem_sequential_12_simple_rnn_5_tensorarrayunstack_tensorlistfromtensor?sequential_12_simple_rnn_5_while_tensorarrayv2read_tensorlistgetitem_sequential_12_simple_rnn_5_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :?????????2: : :::2?
Isequential_12/simple_rnn_5/while/simple_rnn_cell_9/BiasAdd/ReadVariableOpIsequential_12/simple_rnn_5/while/simple_rnn_cell_9/BiasAdd/ReadVariableOp2?
Hsequential_12/simple_rnn_5/while/simple_rnn_cell_9/MatMul/ReadVariableOpHsequential_12/simple_rnn_5/while/simple_rnn_cell_9/MatMul/ReadVariableOp2?
Jsequential_12/simple_rnn_5/while/simple_rnn_cell_9/MatMul_1/ReadVariableOpJsequential_12/simple_rnn_5/while/simple_rnn_cell_9/MatMul_1/ReadVariableOp: 
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

?
simple_rnn_4_while_cond_1009046
2simple_rnn_4_while_simple_rnn_4_while_loop_counter<
8simple_rnn_4_while_simple_rnn_4_while_maximum_iterations"
simple_rnn_4_while_placeholder$
 simple_rnn_4_while_placeholder_1$
 simple_rnn_4_while_placeholder_28
4simple_rnn_4_while_less_simple_rnn_4_strided_slice_1N
Jsimple_rnn_4_while_simple_rnn_4_while_cond_100904___redundant_placeholder0N
Jsimple_rnn_4_while_simple_rnn_4_while_cond_100904___redundant_placeholder1N
Jsimple_rnn_4_while_simple_rnn_4_while_cond_100904___redundant_placeholder2N
Jsimple_rnn_4_while_simple_rnn_4_while_cond_100904___redundant_placeholder3
simple_rnn_4_while_identity
?
simple_rnn_4/while/LessLesssimple_rnn_4_while_placeholder4simple_rnn_4_while_less_simple_rnn_4_strided_slice_1*
T0*
_output_shapes
: 2
simple_rnn_4/while/Less?
simple_rnn_4/while/IdentityIdentitysimple_rnn_4/while/Less:z:0*
T0
*
_output_shapes
: 2
simple_rnn_4/while/Identity"C
simple_rnn_4_while_identity$simple_rnn_4/while/Identity:output:0*A
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
?
J
.__inference_activation_10_layer_call_fn_101634

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
:?????????H?* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_activation_10_layer_call_and_return_conditional_losses_1001092
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:?????????H?2

Identity"
identityIdentity:output:0*+
_input_shapes
:?????????H?:T P
,
_output_shapes
:?????????H?
 
_user_specified_nameinputs
?
e
I__inference_activation_10_layer_call_and_return_conditional_losses_100109

inputs
identityS
ReluReluinputs*
T0*,
_output_shapes
:?????????H?2
Reluk
IdentityIdentityRelu:activations:0*
T0*,
_output_shapes
:?????????H?2

Identity"
identityIdentity:output:0*+
_input_shapes
:?????????H?:T P
,
_output_shapes
:?????????H?
 
_user_specified_nameinputs
?3
?
while_body_101536
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0<
8while_simple_rnn_cell_8_matmul_readvariableop_resource_0=
9while_simple_rnn_cell_8_biasadd_readvariableop_resource_0>
:while_simple_rnn_cell_8_matmul_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor:
6while_simple_rnn_cell_8_matmul_readvariableop_resource;
7while_simple_rnn_cell_8_biasadd_readvariableop_resource<
8while_simple_rnn_cell_8_matmul_1_readvariableop_resource??.while/simple_rnn_cell_8/BiasAdd/ReadVariableOp?-while/simple_rnn_cell_8/MatMul/ReadVariableOp?/while/simple_rnn_cell_8/MatMul_1/ReadVariableOp?
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
-while/simple_rnn_cell_8/MatMul/ReadVariableOpReadVariableOp8while_simple_rnn_cell_8_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype02/
-while/simple_rnn_cell_8/MatMul/ReadVariableOp?
while/simple_rnn_cell_8/MatMul	MLCMatMul0while/TensorArrayV2Read/TensorListGetItem:item:05while/simple_rnn_cell_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2 
while/simple_rnn_cell_8/MatMul?
.while/simple_rnn_cell_8/BiasAdd/ReadVariableOpReadVariableOp9while_simple_rnn_cell_8_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype020
.while/simple_rnn_cell_8/BiasAdd/ReadVariableOp?
while/simple_rnn_cell_8/BiasAddBiasAdd(while/simple_rnn_cell_8/MatMul:product:06while/simple_rnn_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2!
while/simple_rnn_cell_8/BiasAdd?
/while/simple_rnn_cell_8/MatMul_1/ReadVariableOpReadVariableOp:while_simple_rnn_cell_8_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype021
/while/simple_rnn_cell_8/MatMul_1/ReadVariableOp?
 while/simple_rnn_cell_8/MatMul_1	MLCMatMulwhile_placeholder_27while/simple_rnn_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2"
 while/simple_rnn_cell_8/MatMul_1?
while/simple_rnn_cell_8/addAddV2(while/simple_rnn_cell_8/BiasAdd:output:0*while/simple_rnn_cell_8/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
while/simple_rnn_cell_8/add?
while/simple_rnn_cell_8/TanhTanhwhile/simple_rnn_cell_8/add:z:0*
T0*(
_output_shapes
:??????????2
while/simple_rnn_cell_8/Tanh?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder while/simple_rnn_cell_8/Tanh:y:0*
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
while/IdentityIdentitywhile/add_1:z:0/^while/simple_rnn_cell_8/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_8/MatMul/ReadVariableOp0^while/simple_rnn_cell_8/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations/^while/simple_rnn_cell_8/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_8/MatMul/ReadVariableOp0^while/simple_rnn_cell_8/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0/^while/simple_rnn_cell_8/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_8/MatMul/ReadVariableOp0^while/simple_rnn_cell_8/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0/^while/simple_rnn_cell_8/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_8/MatMul/ReadVariableOp0^while/simple_rnn_cell_8/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity while/simple_rnn_cell_8/Tanh:y:0/^while/simple_rnn_cell_8/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_8/MatMul/ReadVariableOp0^while/simple_rnn_cell_8/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2
while/Identity_4")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"t
7while_simple_rnn_cell_8_biasadd_readvariableop_resource9while_simple_rnn_cell_8_biasadd_readvariableop_resource_0"v
8while_simple_rnn_cell_8_matmul_1_readvariableop_resource:while_simple_rnn_cell_8_matmul_1_readvariableop_resource_0"r
6while_simple_rnn_cell_8_matmul_readvariableop_resource8while_simple_rnn_cell_8_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*?
_input_shapes.
,: : : : :??????????: : :::2`
.while/simple_rnn_cell_8/BiasAdd/ReadVariableOp.while/simple_rnn_cell_8/BiasAdd/ReadVariableOp2^
-while/simple_rnn_cell_8/MatMul/ReadVariableOp-while/simple_rnn_cell_8/MatMul/ReadVariableOp2b
/while/simple_rnn_cell_8/MatMul_1/ReadVariableOp/while/simple_rnn_cell_8/MatMul_1/ReadVariableOp: 
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
while_cond_101949
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_101949___redundant_placeholder04
0while_while_cond_101949___redundant_placeholder14
0while_while_cond_101949___redundant_placeholder24
0while_while_cond_101949___redundant_placeholder3
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
??
?
"__inference__traced_restore_102571
file_prefix$
 assignvariableop_dense_14_kernel$
 assignvariableop_1_dense_14_bias 
assignvariableop_2_adam_iter"
assignvariableop_3_adam_beta_1"
assignvariableop_4_adam_beta_2!
assignvariableop_5_adam_decay)
%assignvariableop_6_adam_learning_rate<
8assignvariableop_7_simple_rnn_4_simple_rnn_cell_8_kernelF
Bassignvariableop_8_simple_rnn_4_simple_rnn_cell_8_recurrent_kernel:
6assignvariableop_9_simple_rnn_4_simple_rnn_cell_8_bias=
9assignvariableop_10_simple_rnn_5_simple_rnn_cell_9_kernelG
Cassignvariableop_11_simple_rnn_5_simple_rnn_cell_9_recurrent_kernel;
7assignvariableop_12_simple_rnn_5_simple_rnn_cell_9_bias
assignvariableop_13_total
assignvariableop_14_count
assignvariableop_15_total_1
assignvariableop_16_count_1
assignvariableop_17_total_2
assignvariableop_18_count_2.
*assignvariableop_19_adam_dense_14_kernel_m,
(assignvariableop_20_adam_dense_14_bias_mD
@assignvariableop_21_adam_simple_rnn_4_simple_rnn_cell_8_kernel_mN
Jassignvariableop_22_adam_simple_rnn_4_simple_rnn_cell_8_recurrent_kernel_mB
>assignvariableop_23_adam_simple_rnn_4_simple_rnn_cell_8_bias_mD
@assignvariableop_24_adam_simple_rnn_5_simple_rnn_cell_9_kernel_mN
Jassignvariableop_25_adam_simple_rnn_5_simple_rnn_cell_9_recurrent_kernel_mB
>assignvariableop_26_adam_simple_rnn_5_simple_rnn_cell_9_bias_m.
*assignvariableop_27_adam_dense_14_kernel_v,
(assignvariableop_28_adam_dense_14_bias_vD
@assignvariableop_29_adam_simple_rnn_4_simple_rnn_cell_8_kernel_vN
Jassignvariableop_30_adam_simple_rnn_4_simple_rnn_cell_8_recurrent_kernel_vB
>assignvariableop_31_adam_simple_rnn_4_simple_rnn_cell_8_bias_vD
@assignvariableop_32_adam_simple_rnn_5_simple_rnn_cell_9_kernel_vN
Jassignvariableop_33_adam_simple_rnn_5_simple_rnn_cell_9_recurrent_kernel_vB
>assignvariableop_34_adam_simple_rnn_5_simple_rnn_cell_9_bias_v
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
AssignVariableOpAssignVariableOp assignvariableop_dense_14_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_14_biasIdentity_1:output:0"/device:CPU:0*
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
AssignVariableOp_7AssignVariableOp8assignvariableop_7_simple_rnn_4_simple_rnn_cell_8_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOpBassignvariableop_8_simple_rnn_4_simple_rnn_cell_8_recurrent_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp6assignvariableop_9_simple_rnn_4_simple_rnn_cell_8_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp9assignvariableop_10_simple_rnn_5_simple_rnn_cell_9_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOpCassignvariableop_11_simple_rnn_5_simple_rnn_cell_9_recurrent_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp7assignvariableop_12_simple_rnn_5_simple_rnn_cell_9_biasIdentity_12:output:0"/device:CPU:0*
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
AssignVariableOp_19AssignVariableOp*assignvariableop_19_adam_dense_14_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp(assignvariableop_20_adam_dense_14_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp@assignvariableop_21_adam_simple_rnn_4_simple_rnn_cell_8_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOpJassignvariableop_22_adam_simple_rnn_4_simple_rnn_cell_8_recurrent_kernel_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp>assignvariableop_23_adam_simple_rnn_4_simple_rnn_cell_8_bias_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp@assignvariableop_24_adam_simple_rnn_5_simple_rnn_cell_9_kernel_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOpJassignvariableop_25_adam_simple_rnn_5_simple_rnn_cell_9_recurrent_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp>assignvariableop_26_adam_simple_rnn_5_simple_rnn_cell_9_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp*assignvariableop_27_adam_dense_14_kernel_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp(assignvariableop_28_adam_dense_14_bias_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp@assignvariableop_29_adam_simple_rnn_4_simple_rnn_cell_8_kernel_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOpJassignvariableop_30_adam_simple_rnn_4_simple_rnn_cell_8_recurrent_kernel_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp>assignvariableop_31_adam_simple_rnn_4_simple_rnn_cell_8_bias_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp@assignvariableop_32_adam_simple_rnn_5_simple_rnn_cell_9_kernel_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOpJassignvariableop_33_adam_simple_rnn_5_simple_rnn_cell_9_recurrent_kernel_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp>assignvariableop_34_adam_simple_rnn_5_simple_rnn_cell_9_bias_vIdentity_34:output:0"/device:CPU:0*
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
?
J
.__inference_activation_11_layer_call_fn_102160

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
I__inference_activation_11_layer_call_and_return_conditional_losses_1004022
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
?
?
-__inference_simple_rnn_4_layer_call_fn_101378

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
:?????????H?*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_simple_rnn_4_layer_call_and_return_conditional_losses_1000742
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:?????????H?2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????H:::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????H
 
_user_specified_nameinputs
?H
?
H__inference_simple_rnn_4_layer_call_and_return_conditional_losses_101490
inputs_04
0simple_rnn_cell_8_matmul_readvariableop_resource5
1simple_rnn_cell_8_biasadd_readvariableop_resource6
2simple_rnn_cell_8_matmul_1_readvariableop_resource
identity??(simple_rnn_cell_8/BiasAdd/ReadVariableOp?'simple_rnn_cell_8/MatMul/ReadVariableOp?)simple_rnn_cell_8/MatMul_1/ReadVariableOp?whileF
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
'simple_rnn_cell_8/MatMul/ReadVariableOpReadVariableOp0simple_rnn_cell_8_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02)
'simple_rnn_cell_8/MatMul/ReadVariableOp?
simple_rnn_cell_8/MatMul	MLCMatMulstrided_slice_2:output:0/simple_rnn_cell_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
simple_rnn_cell_8/MatMul?
(simple_rnn_cell_8/BiasAdd/ReadVariableOpReadVariableOp1simple_rnn_cell_8_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02*
(simple_rnn_cell_8/BiasAdd/ReadVariableOp?
simple_rnn_cell_8/BiasAddBiasAdd"simple_rnn_cell_8/MatMul:product:00simple_rnn_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
simple_rnn_cell_8/BiasAdd?
)simple_rnn_cell_8/MatMul_1/ReadVariableOpReadVariableOp2simple_rnn_cell_8_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02+
)simple_rnn_cell_8/MatMul_1/ReadVariableOp?
simple_rnn_cell_8/MatMul_1	MLCMatMulzeros:output:01simple_rnn_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
simple_rnn_cell_8/MatMul_1?
simple_rnn_cell_8/addAddV2"simple_rnn_cell_8/BiasAdd:output:0$simple_rnn_cell_8/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
simple_rnn_cell_8/add?
simple_rnn_cell_8/TanhTanhsimple_rnn_cell_8/add:z:0*
T0*(
_output_shapes
:??????????2
simple_rnn_cell_8/Tanh?
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:00simple_rnn_cell_8_matmul_readvariableop_resource1simple_rnn_cell_8_biasadd_readvariableop_resource2simple_rnn_cell_8_matmul_1_readvariableop_resource*
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
while_body_101424*
condR
while_cond_101423*9
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
IdentityIdentitytranspose_1:y:0)^simple_rnn_cell_8/BiasAdd/ReadVariableOp(^simple_rnn_cell_8/MatMul/ReadVariableOp*^simple_rnn_cell_8/MatMul_1/ReadVariableOp^while*
T0*5
_output_shapes#
!:???????????????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????:::2T
(simple_rnn_cell_8/BiasAdd/ReadVariableOp(simple_rnn_cell_8/BiasAdd/ReadVariableOp2R
'simple_rnn_cell_8/MatMul/ReadVariableOp'simple_rnn_cell_8/MatMul/ReadVariableOp2V
)simple_rnn_cell_8/MatMul_1/ReadVariableOp)simple_rnn_cell_8/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :??????????????????
"
_user_specified_name
inputs/0
?
?
while_cond_101703
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_101703___redundant_placeholder04
0while_while_cond_101703___redundant_placeholder14
0while_while_cond_101703___redundant_placeholder24
0while_while_cond_101703___redundant_placeholder3
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
?
I__inference_sequential_12_layer_call_and_return_conditional_losses_100465
simple_rnn_4_input
simple_rnn_4_100097
simple_rnn_4_100099
simple_rnn_4_100101
simple_rnn_5_100390
simple_rnn_5_100392
simple_rnn_5_100394
dense_14_100459
dense_14_100461
identity?? dense_14/StatefulPartitionedCall?$simple_rnn_4/StatefulPartitionedCall?$simple_rnn_5/StatefulPartitionedCall?
$simple_rnn_4/StatefulPartitionedCallStatefulPartitionedCallsimple_rnn_4_inputsimple_rnn_4_100097simple_rnn_4_100099simple_rnn_4_100101*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????H?*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_simple_rnn_4_layer_call_and_return_conditional_losses_999622&
$simple_rnn_4/StatefulPartitionedCall?
activation_10/PartitionedCallPartitionedCall-simple_rnn_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????H?* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_activation_10_layer_call_and_return_conditional_losses_1001092
activation_10/PartitionedCall?
dropout_4/PartitionedCallPartitionedCall&activation_10/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????H?* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_4_layer_call_and_return_conditional_losses_1001262
dropout_4/PartitionedCall?
$simple_rnn_5/StatefulPartitionedCallStatefulPartitionedCall"dropout_4/PartitionedCall:output:0simple_rnn_5_100390simple_rnn_5_100392simple_rnn_5_100394*
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
H__inference_simple_rnn_5_layer_call_and_return_conditional_losses_1002552&
$simple_rnn_5/StatefulPartitionedCall?
activation_11/PartitionedCallPartitionedCall-simple_rnn_5/StatefulPartitionedCall:output:0*
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
I__inference_activation_11_layer_call_and_return_conditional_losses_1004022
activation_11/PartitionedCall?
dropout_5/PartitionedCallPartitionedCall&activation_11/PartitionedCall:output:0*
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
E__inference_dropout_5_layer_call_and_return_conditional_losses_1004192
dropout_5/PartitionedCall?
 dense_14/StatefulPartitionedCallStatefulPartitionedCall"dropout_5/PartitionedCall:output:0dense_14_100459dense_14_100461*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????$*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_14_layer_call_and_return_conditional_losses_1004482"
 dense_14/StatefulPartitionedCall?
IdentityIdentity)dense_14/StatefulPartitionedCall:output:0!^dense_14/StatefulPartitionedCall%^simple_rnn_4/StatefulPartitionedCall%^simple_rnn_5/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????$2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:?????????H::::::::2D
 dense_14/StatefulPartitionedCall dense_14/StatefulPartitionedCall2L
$simple_rnn_4/StatefulPartitionedCall$simple_rnn_4/StatefulPartitionedCall2L
$simple_rnn_5/StatefulPartitionedCall$simple_rnn_5/StatefulPartitionedCall:_ [
+
_output_shapes
:?????????H
,
_user_specified_namesimple_rnn_4_input
?#
?
while_body_99146
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0#
while_simple_rnn_cell_8_99168_0#
while_simple_rnn_cell_8_99170_0#
while_simple_rnn_cell_8_99172_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor!
while_simple_rnn_cell_8_99168!
while_simple_rnn_cell_8_99170!
while_simple_rnn_cell_8_99172??/while/simple_rnn_cell_8/StatefulPartitionedCall?
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
/while/simple_rnn_cell_8/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_simple_rnn_cell_8_99168_0while_simple_rnn_cell_8_99170_0while_simple_rnn_cell_8_99172_0*
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
GPU 2J 8? *U
fPRN
L__inference_simple_rnn_cell_8_layer_call_and_return_conditional_losses_9887221
/while/simple_rnn_cell_8/StatefulPartitionedCall?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder8while/simple_rnn_cell_8/StatefulPartitionedCall:output:0*
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
while/IdentityIdentitywhile/add_1:z:00^while/simple_rnn_cell_8/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations0^while/simple_rnn_cell_8/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:00^while/simple_rnn_cell_8/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:00^while/simple_rnn_cell_8/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity8while/simple_rnn_cell_8/StatefulPartitionedCall:output:10^while/simple_rnn_cell_8/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2
while/Identity_4")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"@
while_simple_rnn_cell_8_99168while_simple_rnn_cell_8_99168_0"@
while_simple_rnn_cell_8_99170while_simple_rnn_cell_8_99170_0"@
while_simple_rnn_cell_8_99172while_simple_rnn_cell_8_99172_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*?
_input_shapes.
,: : : : :??????????: : :::2b
/while/simple_rnn_cell_8/StatefulPartitionedCall/while/simple_rnn_cell_8/StatefulPartitionedCall: 
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
while_cond_99774
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_99774___redundant_placeholder03
/while_while_cond_99774___redundant_placeholder13
/while_while_cond_99774___redundant_placeholder23
/while_while_cond_99774___redundant_placeholder3
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
-__inference_simple_rnn_5_layer_call_fn_101904
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
GPU 2J 8? *P
fKRI
G__inference_simple_rnn_5_layer_call_and_return_conditional_losses_998382
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
?
?
.__inference_sequential_12_layer_call_fn_100541
simple_rnn_4_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallsimple_rnn_4_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????$**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_sequential_12_layer_call_and_return_conditional_losses_1005222
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????$2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:?????????H::::::::22
StatefulPartitionedCallStatefulPartitionedCall:_ [
+
_output_shapes
:?????????H
,
_user_specified_namesimple_rnn_4_input
?

?
simple_rnn_5_while_cond_1007796
2simple_rnn_5_while_simple_rnn_5_while_loop_counter<
8simple_rnn_5_while_simple_rnn_5_while_maximum_iterations"
simple_rnn_5_while_placeholder$
 simple_rnn_5_while_placeholder_1$
 simple_rnn_5_while_placeholder_28
4simple_rnn_5_while_less_simple_rnn_5_strided_slice_1N
Jsimple_rnn_5_while_simple_rnn_5_while_cond_100779___redundant_placeholder0N
Jsimple_rnn_5_while_simple_rnn_5_while_cond_100779___redundant_placeholder1N
Jsimple_rnn_5_while_simple_rnn_5_while_cond_100779___redundant_placeholder2N
Jsimple_rnn_5_while_simple_rnn_5_while_cond_100779___redundant_placeholder3
simple_rnn_5_while_identity
?
simple_rnn_5/while/LessLesssimple_rnn_5_while_placeholder4simple_rnn_5_while_less_simple_rnn_5_strided_slice_1*
T0*
_output_shapes
: 2
simple_rnn_5/while/Less?
simple_rnn_5/while/IdentityIdentitysimple_rnn_5/while/Less:z:0*
T0
*
_output_shapes
: 2
simple_rnn_5/while/Identity"C
simple_rnn_5_while_identity$simple_rnn_5/while/Identity:output:0*@
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
?
e
I__inference_activation_10_layer_call_and_return_conditional_losses_101629

inputs
identityS
ReluReluinputs*
T0*,
_output_shapes
:?????????H?2
Reluk
IdentityIdentityRelu:activations:0*
T0*,
_output_shapes
:?????????H?2

Identity"
identityIdentity:output:0*+
_input_shapes
:?????????H?:T P
,
_output_shapes
:?????????H?
 
_user_specified_nameinputs
?
F
*__inference_dropout_4_layer_call_fn_101658

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
:?????????H?* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_4_layer_call_and_return_conditional_losses_1001312
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:?????????H?2

Identity"
identityIdentity:output:0*+
_input_shapes
:?????????H?:T P
,
_output_shapes
:?????????H?
 
_user_specified_nameinputs
?
?
while_cond_102061
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_102061___redundant_placeholder04
0while_while_cond_102061___redundant_placeholder14
0while_while_cond_102061___redundant_placeholder24
0while_while_cond_102061___redundant_placeholder3
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
?
?
M__inference_simple_rnn_cell_8_layer_call_and_return_conditional_losses_102221

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
?
?
while_cond_101423
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_101423___redundant_placeholder04
0while_while_cond_101423___redundant_placeholder14
0while_while_cond_101423___redundant_placeholder24
0while_while_cond_101423___redundant_placeholder3
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
simple_rnn_5_while_cond_1010146
2simple_rnn_5_while_simple_rnn_5_while_loop_counter<
8simple_rnn_5_while_simple_rnn_5_while_maximum_iterations"
simple_rnn_5_while_placeholder$
 simple_rnn_5_while_placeholder_1$
 simple_rnn_5_while_placeholder_28
4simple_rnn_5_while_less_simple_rnn_5_strided_slice_1N
Jsimple_rnn_5_while_simple_rnn_5_while_cond_101014___redundant_placeholder0N
Jsimple_rnn_5_while_simple_rnn_5_while_cond_101014___redundant_placeholder1N
Jsimple_rnn_5_while_simple_rnn_5_while_cond_101014___redundant_placeholder2N
Jsimple_rnn_5_while_simple_rnn_5_while_cond_101014___redundant_placeholder3
simple_rnn_5_while_identity
?
simple_rnn_5/while/LessLesssimple_rnn_5_while_placeholder4simple_rnn_5_while_less_simple_rnn_5_strided_slice_1*
T0*
_output_shapes
: 2
simple_rnn_5/while/Less?
simple_rnn_5/while/IdentityIdentitysimple_rnn_5/while/Less:z:0*
T0
*
_output_shapes
: 2
simple_rnn_5/while/Identity"C
simple_rnn_5_while_identity$simple_rnn_5/while/Identity:output:0*@
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
?
I__inference_sequential_12_layer_call_and_return_conditional_losses_100570

inputs
simple_rnn_4_100546
simple_rnn_4_100548
simple_rnn_4_100550
simple_rnn_5_100555
simple_rnn_5_100557
simple_rnn_5_100559
dense_14_100564
dense_14_100566
identity?? dense_14/StatefulPartitionedCall?$simple_rnn_4/StatefulPartitionedCall?$simple_rnn_5/StatefulPartitionedCall?
$simple_rnn_4/StatefulPartitionedCallStatefulPartitionedCallinputssimple_rnn_4_100546simple_rnn_4_100548simple_rnn_4_100550*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????H?*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_simple_rnn_4_layer_call_and_return_conditional_losses_1000742&
$simple_rnn_4/StatefulPartitionedCall?
activation_10/PartitionedCallPartitionedCall-simple_rnn_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????H?* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_activation_10_layer_call_and_return_conditional_losses_1001092
activation_10/PartitionedCall?
dropout_4/PartitionedCallPartitionedCall&activation_10/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????H?* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_4_layer_call_and_return_conditional_losses_1001312
dropout_4/PartitionedCall?
$simple_rnn_5/StatefulPartitionedCallStatefulPartitionedCall"dropout_4/PartitionedCall:output:0simple_rnn_5_100555simple_rnn_5_100557simple_rnn_5_100559*
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
H__inference_simple_rnn_5_layer_call_and_return_conditional_losses_1003672&
$simple_rnn_5/StatefulPartitionedCall?
activation_11/PartitionedCallPartitionedCall-simple_rnn_5/StatefulPartitionedCall:output:0*
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
I__inference_activation_11_layer_call_and_return_conditional_losses_1004022
activation_11/PartitionedCall?
dropout_5/PartitionedCallPartitionedCall&activation_11/PartitionedCall:output:0*
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
E__inference_dropout_5_layer_call_and_return_conditional_losses_1004242
dropout_5/PartitionedCall?
 dense_14/StatefulPartitionedCallStatefulPartitionedCall"dropout_5/PartitionedCall:output:0dense_14_100564dense_14_100566*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????$*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_14_layer_call_and_return_conditional_losses_1004482"
 dense_14/StatefulPartitionedCall?
IdentityIdentity)dense_14/StatefulPartitionedCall:output:0!^dense_14/StatefulPartitionedCall%^simple_rnn_4/StatefulPartitionedCall%^simple_rnn_5/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????$2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:?????????H::::::::2D
 dense_14/StatefulPartitionedCall dense_14/StatefulPartitionedCall2L
$simple_rnn_4/StatefulPartitionedCall$simple_rnn_4/StatefulPartitionedCall2L
$simple_rnn_5/StatefulPartitionedCall$simple_rnn_5/StatefulPartitionedCall:S O
+
_output_shapes
:?????????H
 
_user_specified_nameinputs
?
?
-__inference_simple_rnn_4_layer_call_fn_101367

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
:?????????H?*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_simple_rnn_4_layer_call_and_return_conditional_losses_999622
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:?????????H?2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????H:::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????H
 
_user_specified_nameinputs
?
c
E__inference_dropout_5_layer_call_and_return_conditional_losses_102174

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
?
c
E__inference_dropout_4_layer_call_and_return_conditional_losses_100131

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:?????????H?2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:?????????H?2

Identity_1"!

identity_1Identity_1:output:0*+
_input_shapes
:?????????H?:T P
,
_output_shapes
:?????????H?
 
_user_specified_nameinputs
?
?
.__inference_sequential_12_layer_call_fn_101132

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
:?????????$**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_sequential_12_layer_call_and_return_conditional_losses_1005702
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????$2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:?????????H::::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????H
 
_user_specified_nameinputs
?
?
while_cond_101289
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_101289___redundant_placeholder04
0while_while_cond_101289___redundant_placeholder14
0while_while_cond_101289___redundant_placeholder24
0while_while_cond_101289___redundant_placeholder3
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
?B
?
simple_rnn_5_while_body_1010156
2simple_rnn_5_while_simple_rnn_5_while_loop_counter<
8simple_rnn_5_while_simple_rnn_5_while_maximum_iterations"
simple_rnn_5_while_placeholder$
 simple_rnn_5_while_placeholder_1$
 simple_rnn_5_while_placeholder_25
1simple_rnn_5_while_simple_rnn_5_strided_slice_1_0q
msimple_rnn_5_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_5_tensorarrayunstack_tensorlistfromtensor_0I
Esimple_rnn_5_while_simple_rnn_cell_9_matmul_readvariableop_resource_0J
Fsimple_rnn_5_while_simple_rnn_cell_9_biasadd_readvariableop_resource_0K
Gsimple_rnn_5_while_simple_rnn_cell_9_matmul_1_readvariableop_resource_0
simple_rnn_5_while_identity!
simple_rnn_5_while_identity_1!
simple_rnn_5_while_identity_2!
simple_rnn_5_while_identity_3!
simple_rnn_5_while_identity_43
/simple_rnn_5_while_simple_rnn_5_strided_slice_1o
ksimple_rnn_5_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_5_tensorarrayunstack_tensorlistfromtensorG
Csimple_rnn_5_while_simple_rnn_cell_9_matmul_readvariableop_resourceH
Dsimple_rnn_5_while_simple_rnn_cell_9_biasadd_readvariableop_resourceI
Esimple_rnn_5_while_simple_rnn_cell_9_matmul_1_readvariableop_resource??;simple_rnn_5/while/simple_rnn_cell_9/BiasAdd/ReadVariableOp?:simple_rnn_5/while/simple_rnn_cell_9/MatMul/ReadVariableOp?<simple_rnn_5/while/simple_rnn_cell_9/MatMul_1/ReadVariableOp?
Dsimple_rnn_5/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2F
Dsimple_rnn_5/while/TensorArrayV2Read/TensorListGetItem/element_shape?
6simple_rnn_5/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemmsimple_rnn_5_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_5_tensorarrayunstack_tensorlistfromtensor_0simple_rnn_5_while_placeholderMsimple_rnn_5/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype028
6simple_rnn_5/while/TensorArrayV2Read/TensorListGetItem?
:simple_rnn_5/while/simple_rnn_cell_9/MatMul/ReadVariableOpReadVariableOpEsimple_rnn_5_while_simple_rnn_cell_9_matmul_readvariableop_resource_0*
_output_shapes
:	?2*
dtype02<
:simple_rnn_5/while/simple_rnn_cell_9/MatMul/ReadVariableOp?
+simple_rnn_5/while/simple_rnn_cell_9/MatMul	MLCMatMul=simple_rnn_5/while/TensorArrayV2Read/TensorListGetItem:item:0Bsimple_rnn_5/while/simple_rnn_cell_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22-
+simple_rnn_5/while/simple_rnn_cell_9/MatMul?
;simple_rnn_5/while/simple_rnn_cell_9/BiasAdd/ReadVariableOpReadVariableOpFsimple_rnn_5_while_simple_rnn_cell_9_biasadd_readvariableop_resource_0*
_output_shapes
:2*
dtype02=
;simple_rnn_5/while/simple_rnn_cell_9/BiasAdd/ReadVariableOp?
,simple_rnn_5/while/simple_rnn_cell_9/BiasAddBiasAdd5simple_rnn_5/while/simple_rnn_cell_9/MatMul:product:0Csimple_rnn_5/while/simple_rnn_cell_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22.
,simple_rnn_5/while/simple_rnn_cell_9/BiasAdd?
<simple_rnn_5/while/simple_rnn_cell_9/MatMul_1/ReadVariableOpReadVariableOpGsimple_rnn_5_while_simple_rnn_cell_9_matmul_1_readvariableop_resource_0*
_output_shapes

:22*
dtype02>
<simple_rnn_5/while/simple_rnn_cell_9/MatMul_1/ReadVariableOp?
-simple_rnn_5/while/simple_rnn_cell_9/MatMul_1	MLCMatMul simple_rnn_5_while_placeholder_2Dsimple_rnn_5/while/simple_rnn_cell_9/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22/
-simple_rnn_5/while/simple_rnn_cell_9/MatMul_1?
(simple_rnn_5/while/simple_rnn_cell_9/addAddV25simple_rnn_5/while/simple_rnn_cell_9/BiasAdd:output:07simple_rnn_5/while/simple_rnn_cell_9/MatMul_1:product:0*
T0*'
_output_shapes
:?????????22*
(simple_rnn_5/while/simple_rnn_cell_9/add?
)simple_rnn_5/while/simple_rnn_cell_9/TanhTanh,simple_rnn_5/while/simple_rnn_cell_9/add:z:0*
T0*'
_output_shapes
:?????????22+
)simple_rnn_5/while/simple_rnn_cell_9/Tanh?
7simple_rnn_5/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem simple_rnn_5_while_placeholder_1simple_rnn_5_while_placeholder-simple_rnn_5/while/simple_rnn_cell_9/Tanh:y:0*
_output_shapes
: *
element_dtype029
7simple_rnn_5/while/TensorArrayV2Write/TensorListSetItemv
simple_rnn_5/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
simple_rnn_5/while/add/y?
simple_rnn_5/while/addAddV2simple_rnn_5_while_placeholder!simple_rnn_5/while/add/y:output:0*
T0*
_output_shapes
: 2
simple_rnn_5/while/addz
simple_rnn_5/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
simple_rnn_5/while/add_1/y?
simple_rnn_5/while/add_1AddV22simple_rnn_5_while_simple_rnn_5_while_loop_counter#simple_rnn_5/while/add_1/y:output:0*
T0*
_output_shapes
: 2
simple_rnn_5/while/add_1?
simple_rnn_5/while/IdentityIdentitysimple_rnn_5/while/add_1:z:0<^simple_rnn_5/while/simple_rnn_cell_9/BiasAdd/ReadVariableOp;^simple_rnn_5/while/simple_rnn_cell_9/MatMul/ReadVariableOp=^simple_rnn_5/while/simple_rnn_cell_9/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
simple_rnn_5/while/Identity?
simple_rnn_5/while/Identity_1Identity8simple_rnn_5_while_simple_rnn_5_while_maximum_iterations<^simple_rnn_5/while/simple_rnn_cell_9/BiasAdd/ReadVariableOp;^simple_rnn_5/while/simple_rnn_cell_9/MatMul/ReadVariableOp=^simple_rnn_5/while/simple_rnn_cell_9/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
simple_rnn_5/while/Identity_1?
simple_rnn_5/while/Identity_2Identitysimple_rnn_5/while/add:z:0<^simple_rnn_5/while/simple_rnn_cell_9/BiasAdd/ReadVariableOp;^simple_rnn_5/while/simple_rnn_cell_9/MatMul/ReadVariableOp=^simple_rnn_5/while/simple_rnn_cell_9/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
simple_rnn_5/while/Identity_2?
simple_rnn_5/while/Identity_3IdentityGsimple_rnn_5/while/TensorArrayV2Write/TensorListSetItem:output_handle:0<^simple_rnn_5/while/simple_rnn_cell_9/BiasAdd/ReadVariableOp;^simple_rnn_5/while/simple_rnn_cell_9/MatMul/ReadVariableOp=^simple_rnn_5/while/simple_rnn_cell_9/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
simple_rnn_5/while/Identity_3?
simple_rnn_5/while/Identity_4Identity-simple_rnn_5/while/simple_rnn_cell_9/Tanh:y:0<^simple_rnn_5/while/simple_rnn_cell_9/BiasAdd/ReadVariableOp;^simple_rnn_5/while/simple_rnn_cell_9/MatMul/ReadVariableOp=^simple_rnn_5/while/simple_rnn_cell_9/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????22
simple_rnn_5/while/Identity_4"C
simple_rnn_5_while_identity$simple_rnn_5/while/Identity:output:0"G
simple_rnn_5_while_identity_1&simple_rnn_5/while/Identity_1:output:0"G
simple_rnn_5_while_identity_2&simple_rnn_5/while/Identity_2:output:0"G
simple_rnn_5_while_identity_3&simple_rnn_5/while/Identity_3:output:0"G
simple_rnn_5_while_identity_4&simple_rnn_5/while/Identity_4:output:0"d
/simple_rnn_5_while_simple_rnn_5_strided_slice_11simple_rnn_5_while_simple_rnn_5_strided_slice_1_0"?
Dsimple_rnn_5_while_simple_rnn_cell_9_biasadd_readvariableop_resourceFsimple_rnn_5_while_simple_rnn_cell_9_biasadd_readvariableop_resource_0"?
Esimple_rnn_5_while_simple_rnn_cell_9_matmul_1_readvariableop_resourceGsimple_rnn_5_while_simple_rnn_cell_9_matmul_1_readvariableop_resource_0"?
Csimple_rnn_5_while_simple_rnn_cell_9_matmul_readvariableop_resourceEsimple_rnn_5_while_simple_rnn_cell_9_matmul_readvariableop_resource_0"?
ksimple_rnn_5_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_5_tensorarrayunstack_tensorlistfromtensormsimple_rnn_5_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_5_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :?????????2: : :::2z
;simple_rnn_5/while/simple_rnn_cell_9/BiasAdd/ReadVariableOp;simple_rnn_5/while/simple_rnn_cell_9/BiasAdd/ReadVariableOp2x
:simple_rnn_5/while/simple_rnn_cell_9/MatMul/ReadVariableOp:simple_rnn_5/while/simple_rnn_cell_9/MatMul/ReadVariableOp2|
<simple_rnn_5/while/simple_rnn_cell_9/MatMul_1/ReadVariableOp<simple_rnn_5/while/simple_rnn_cell_9/MatMul_1/ReadVariableOp: 
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
?
while_cond_101535
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_101535___redundant_placeholder04
0while_while_cond_101535___redundant_placeholder14
0while_while_cond_101535___redundant_placeholder24
0while_while_cond_101535___redundant_placeholder3
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

?
D__inference_dense_14_layer_call_and_return_conditional_losses_102195

inputs%
!mlcmatmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MLCMatMul/ReadVariableOp?
MLCMatMul/ReadVariableOpReadVariableOp!mlcmatmul_readvariableop_resource*
_output_shapes

:2$*
dtype02
MLCMatMul/ReadVariableOp
	MLCMatMul	MLCMatMulinputs MLCMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????$2
	MLCMatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:$*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMLCMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????$2	
BiasAddX
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:?????????$2
Tanh?
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MLCMatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????$2

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
?	
?
2__inference_simple_rnn_cell_9_layer_call_fn_102328

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
GPU 2J 8? *U
fPRN
L__inference_simple_rnn_cell_9_layer_call_and_return_conditional_losses_994012
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
?

?
D__inference_dense_14_layer_call_and_return_conditional_losses_100448

inputs%
!mlcmatmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MLCMatMul/ReadVariableOp?
MLCMatMul/ReadVariableOpReadVariableOp!mlcmatmul_readvariableop_resource*
_output_shapes

:2$*
dtype02
MLCMatMul/ReadVariableOp
	MLCMatMul	MLCMatMulinputs MLCMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????$2
	MLCMatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:$*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMLCMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????$2	
BiasAddX
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:?????????$2
Tanh?
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MLCMatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????$2

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
?#
?
while_body_99658
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0#
while_simple_rnn_cell_9_99680_0#
while_simple_rnn_cell_9_99682_0#
while_simple_rnn_cell_9_99684_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor!
while_simple_rnn_cell_9_99680!
while_simple_rnn_cell_9_99682!
while_simple_rnn_cell_9_99684??/while/simple_rnn_cell_9/StatefulPartitionedCall?
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
/while/simple_rnn_cell_9/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_simple_rnn_cell_9_99680_0while_simple_rnn_cell_9_99682_0while_simple_rnn_cell_9_99684_0*
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
GPU 2J 8? *U
fPRN
L__inference_simple_rnn_cell_9_layer_call_and_return_conditional_losses_9938421
/while/simple_rnn_cell_9/StatefulPartitionedCall?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder8while/simple_rnn_cell_9/StatefulPartitionedCall:output:0*
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
while/IdentityIdentitywhile/add_1:z:00^while/simple_rnn_cell_9/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations0^while/simple_rnn_cell_9/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:00^while/simple_rnn_cell_9/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:00^while/simple_rnn_cell_9/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity8while/simple_rnn_cell_9/StatefulPartitionedCall:output:10^while/simple_rnn_cell_9/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????22
while/Identity_4")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"@
while_simple_rnn_cell_9_99680while_simple_rnn_cell_9_99680_0"@
while_simple_rnn_cell_9_99682while_simple_rnn_cell_9_99682_0"@
while_simple_rnn_cell_9_99684while_simple_rnn_cell_9_99684_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :?????????2: : :::2b
/while/simple_rnn_cell_9/StatefulPartitionedCall/while/simple_rnn_cell_9/StatefulPartitionedCall: 
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
?
?
M__inference_simple_rnn_cell_8_layer_call_and_return_conditional_losses_102238

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
?
~
)__inference_dense_14_layer_call_fn_102204

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
:?????????$*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_14_layer_call_and_return_conditional_losses_1004482
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????$2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????2::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs
?<
?
G__inference_simple_rnn_4_layer_call_and_return_conditional_losses_99326

inputs
simple_rnn_cell_8_99251
simple_rnn_cell_8_99253
simple_rnn_cell_8_99255
identity??)simple_rnn_cell_8/StatefulPartitionedCall?whileD
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
)simple_rnn_cell_8/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0simple_rnn_cell_8_99251simple_rnn_cell_8_99253simple_rnn_cell_8_99255*
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
GPU 2J 8? *U
fPRN
L__inference_simple_rnn_cell_8_layer_call_and_return_conditional_losses_988892+
)simple_rnn_cell_8/StatefulPartitionedCall?
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0simple_rnn_cell_8_99251simple_rnn_cell_8_99253simple_rnn_cell_8_99255*
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
bodyR
while_body_99263*
condR
while_cond_99262*9
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
IdentityIdentitytranspose_1:y:0*^simple_rnn_cell_8/StatefulPartitionedCall^while*
T0*5
_output_shapes#
!:???????????????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????:::2V
)simple_rnn_cell_8/StatefulPartitionedCall)simple_rnn_cell_8/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?3
?
while_body_99896
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0<
8while_simple_rnn_cell_8_matmul_readvariableop_resource_0=
9while_simple_rnn_cell_8_biasadd_readvariableop_resource_0>
:while_simple_rnn_cell_8_matmul_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor:
6while_simple_rnn_cell_8_matmul_readvariableop_resource;
7while_simple_rnn_cell_8_biasadd_readvariableop_resource<
8while_simple_rnn_cell_8_matmul_1_readvariableop_resource??.while/simple_rnn_cell_8/BiasAdd/ReadVariableOp?-while/simple_rnn_cell_8/MatMul/ReadVariableOp?/while/simple_rnn_cell_8/MatMul_1/ReadVariableOp?
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
-while/simple_rnn_cell_8/MatMul/ReadVariableOpReadVariableOp8while_simple_rnn_cell_8_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype02/
-while/simple_rnn_cell_8/MatMul/ReadVariableOp?
while/simple_rnn_cell_8/MatMul	MLCMatMul0while/TensorArrayV2Read/TensorListGetItem:item:05while/simple_rnn_cell_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2 
while/simple_rnn_cell_8/MatMul?
.while/simple_rnn_cell_8/BiasAdd/ReadVariableOpReadVariableOp9while_simple_rnn_cell_8_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype020
.while/simple_rnn_cell_8/BiasAdd/ReadVariableOp?
while/simple_rnn_cell_8/BiasAddBiasAdd(while/simple_rnn_cell_8/MatMul:product:06while/simple_rnn_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2!
while/simple_rnn_cell_8/BiasAdd?
/while/simple_rnn_cell_8/MatMul_1/ReadVariableOpReadVariableOp:while_simple_rnn_cell_8_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype021
/while/simple_rnn_cell_8/MatMul_1/ReadVariableOp?
 while/simple_rnn_cell_8/MatMul_1	MLCMatMulwhile_placeholder_27while/simple_rnn_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2"
 while/simple_rnn_cell_8/MatMul_1?
while/simple_rnn_cell_8/addAddV2(while/simple_rnn_cell_8/BiasAdd:output:0*while/simple_rnn_cell_8/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
while/simple_rnn_cell_8/add?
while/simple_rnn_cell_8/TanhTanhwhile/simple_rnn_cell_8/add:z:0*
T0*(
_output_shapes
:??????????2
while/simple_rnn_cell_8/Tanh?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder while/simple_rnn_cell_8/Tanh:y:0*
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
while/IdentityIdentitywhile/add_1:z:0/^while/simple_rnn_cell_8/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_8/MatMul/ReadVariableOp0^while/simple_rnn_cell_8/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations/^while/simple_rnn_cell_8/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_8/MatMul/ReadVariableOp0^while/simple_rnn_cell_8/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0/^while/simple_rnn_cell_8/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_8/MatMul/ReadVariableOp0^while/simple_rnn_cell_8/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0/^while/simple_rnn_cell_8/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_8/MatMul/ReadVariableOp0^while/simple_rnn_cell_8/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity while/simple_rnn_cell_8/Tanh:y:0/^while/simple_rnn_cell_8/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_8/MatMul/ReadVariableOp0^while/simple_rnn_cell_8/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2
while/Identity_4")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"t
7while_simple_rnn_cell_8_biasadd_readvariableop_resource9while_simple_rnn_cell_8_biasadd_readvariableop_resource_0"v
8while_simple_rnn_cell_8_matmul_1_readvariableop_resource:while_simple_rnn_cell_8_matmul_1_readvariableop_resource_0"r
6while_simple_rnn_cell_8_matmul_readvariableop_resource8while_simple_rnn_cell_8_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*?
_input_shapes.
,: : : : :??????????: : :::2`
.while/simple_rnn_cell_8/BiasAdd/ReadVariableOp.while/simple_rnn_cell_8/BiasAdd/ReadVariableOp2^
-while/simple_rnn_cell_8/MatMul/ReadVariableOp-while/simple_rnn_cell_8/MatMul/ReadVariableOp2b
/while/simple_rnn_cell_8/MatMul_1/ReadVariableOp/while/simple_rnn_cell_8/MatMul_1/ReadVariableOp: 
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
e
I__inference_activation_11_layer_call_and_return_conditional_losses_100402

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
E__inference_dropout_4_layer_call_and_return_conditional_losses_101648

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:?????????H?2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:?????????H?2

Identity_1"!

identity_1Identity_1:output:0*+
_input_shapes
:?????????H?:T P
,
_output_shapes
:?????????H?
 
_user_specified_nameinputs
?G
?
H__inference_simple_rnn_5_layer_call_and_return_conditional_losses_102128

inputs4
0simple_rnn_cell_9_matmul_readvariableop_resource5
1simple_rnn_cell_9_biasadd_readvariableop_resource6
2simple_rnn_cell_9_matmul_1_readvariableop_resource
identity??(simple_rnn_cell_9/BiasAdd/ReadVariableOp?'simple_rnn_cell_9/MatMul/ReadVariableOp?)simple_rnn_cell_9/MatMul_1/ReadVariableOp?whileD
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
:H??????????2
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
'simple_rnn_cell_9/MatMul/ReadVariableOpReadVariableOp0simple_rnn_cell_9_matmul_readvariableop_resource*
_output_shapes
:	?2*
dtype02)
'simple_rnn_cell_9/MatMul/ReadVariableOp?
simple_rnn_cell_9/MatMul	MLCMatMulstrided_slice_2:output:0/simple_rnn_cell_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
simple_rnn_cell_9/MatMul?
(simple_rnn_cell_9/BiasAdd/ReadVariableOpReadVariableOp1simple_rnn_cell_9_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02*
(simple_rnn_cell_9/BiasAdd/ReadVariableOp?
simple_rnn_cell_9/BiasAddBiasAdd"simple_rnn_cell_9/MatMul:product:00simple_rnn_cell_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
simple_rnn_cell_9/BiasAdd?
)simple_rnn_cell_9/MatMul_1/ReadVariableOpReadVariableOp2simple_rnn_cell_9_matmul_1_readvariableop_resource*
_output_shapes

:22*
dtype02+
)simple_rnn_cell_9/MatMul_1/ReadVariableOp?
simple_rnn_cell_9/MatMul_1	MLCMatMulzeros:output:01simple_rnn_cell_9/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
simple_rnn_cell_9/MatMul_1?
simple_rnn_cell_9/addAddV2"simple_rnn_cell_9/BiasAdd:output:0$simple_rnn_cell_9/MatMul_1:product:0*
T0*'
_output_shapes
:?????????22
simple_rnn_cell_9/add?
simple_rnn_cell_9/TanhTanhsimple_rnn_cell_9/add:z:0*
T0*'
_output_shapes
:?????????22
simple_rnn_cell_9/Tanh?
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:00simple_rnn_cell_9_matmul_readvariableop_resource1simple_rnn_cell_9_biasadd_readvariableop_resource2simple_rnn_cell_9_matmul_1_readvariableop_resource*
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
while_body_102062*
condR
while_cond_102061*8
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
:H?????????2*
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
:?????????H22
transpose_1?
IdentityIdentitystrided_slice_3:output:0)^simple_rnn_cell_9/BiasAdd/ReadVariableOp(^simple_rnn_cell_9/MatMul/ReadVariableOp*^simple_rnn_cell_9/MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????H?:::2T
(simple_rnn_cell_9/BiasAdd/ReadVariableOp(simple_rnn_cell_9/BiasAdd/ReadVariableOp2R
'simple_rnn_cell_9/MatMul/ReadVariableOp'simple_rnn_cell_9/MatMul/ReadVariableOp2V
)simple_rnn_cell_9/MatMul_1/ReadVariableOp)simple_rnn_cell_9/MatMul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:?????????H?
 
_user_specified_nameinputs
?H
?
H__inference_simple_rnn_5_layer_call_and_return_conditional_losses_101882
inputs_04
0simple_rnn_cell_9_matmul_readvariableop_resource5
1simple_rnn_cell_9_biasadd_readvariableop_resource6
2simple_rnn_cell_9_matmul_1_readvariableop_resource
identity??(simple_rnn_cell_9/BiasAdd/ReadVariableOp?'simple_rnn_cell_9/MatMul/ReadVariableOp?)simple_rnn_cell_9/MatMul_1/ReadVariableOp?whileF
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
'simple_rnn_cell_9/MatMul/ReadVariableOpReadVariableOp0simple_rnn_cell_9_matmul_readvariableop_resource*
_output_shapes
:	?2*
dtype02)
'simple_rnn_cell_9/MatMul/ReadVariableOp?
simple_rnn_cell_9/MatMul	MLCMatMulstrided_slice_2:output:0/simple_rnn_cell_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
simple_rnn_cell_9/MatMul?
(simple_rnn_cell_9/BiasAdd/ReadVariableOpReadVariableOp1simple_rnn_cell_9_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02*
(simple_rnn_cell_9/BiasAdd/ReadVariableOp?
simple_rnn_cell_9/BiasAddBiasAdd"simple_rnn_cell_9/MatMul:product:00simple_rnn_cell_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
simple_rnn_cell_9/BiasAdd?
)simple_rnn_cell_9/MatMul_1/ReadVariableOpReadVariableOp2simple_rnn_cell_9_matmul_1_readvariableop_resource*
_output_shapes

:22*
dtype02+
)simple_rnn_cell_9/MatMul_1/ReadVariableOp?
simple_rnn_cell_9/MatMul_1	MLCMatMulzeros:output:01simple_rnn_cell_9/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
simple_rnn_cell_9/MatMul_1?
simple_rnn_cell_9/addAddV2"simple_rnn_cell_9/BiasAdd:output:0$simple_rnn_cell_9/MatMul_1:product:0*
T0*'
_output_shapes
:?????????22
simple_rnn_cell_9/add?
simple_rnn_cell_9/TanhTanhsimple_rnn_cell_9/add:z:0*
T0*'
_output_shapes
:?????????22
simple_rnn_cell_9/Tanh?
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:00simple_rnn_cell_9_matmul_readvariableop_resource1simple_rnn_cell_9_biasadd_readvariableop_resource2simple_rnn_cell_9_matmul_1_readvariableop_resource*
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
while_body_101816*
condR
while_cond_101815*8
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
IdentityIdentitystrided_slice_3:output:0)^simple_rnn_cell_9/BiasAdd/ReadVariableOp(^simple_rnn_cell_9/MatMul/ReadVariableOp*^simple_rnn_cell_9/MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*@
_input_shapes/
-:???????????????????:::2T
(simple_rnn_cell_9/BiasAdd/ReadVariableOp(simple_rnn_cell_9/BiasAdd/ReadVariableOp2R
'simple_rnn_cell_9/MatMul/ReadVariableOp'simple_rnn_cell_9/MatMul/ReadVariableOp2V
)simple_rnn_cell_9/MatMul_1/ReadVariableOp)simple_rnn_cell_9/MatMul_1/ReadVariableOp2
whilewhile:_ [
5
_output_shapes#
!:???????????????????
"
_user_specified_name
inputs/0
??
?
I__inference_sequential_12_layer_call_and_return_conditional_losses_100859

inputsA
=simple_rnn_4_simple_rnn_cell_8_matmul_readvariableop_resourceB
>simple_rnn_4_simple_rnn_cell_8_biasadd_readvariableop_resourceC
?simple_rnn_4_simple_rnn_cell_8_matmul_1_readvariableop_resourceA
=simple_rnn_5_simple_rnn_cell_9_matmul_readvariableop_resourceB
>simple_rnn_5_simple_rnn_cell_9_biasadd_readvariableop_resourceC
?simple_rnn_5_simple_rnn_cell_9_matmul_1_readvariableop_resource.
*dense_14_mlcmatmul_readvariableop_resource,
(dense_14_biasadd_readvariableop_resource
identity??dense_14/BiasAdd/ReadVariableOp?!dense_14/MLCMatMul/ReadVariableOp?5simple_rnn_4/simple_rnn_cell_8/BiasAdd/ReadVariableOp?4simple_rnn_4/simple_rnn_cell_8/MatMul/ReadVariableOp?6simple_rnn_4/simple_rnn_cell_8/MatMul_1/ReadVariableOp?simple_rnn_4/while?5simple_rnn_5/simple_rnn_cell_9/BiasAdd/ReadVariableOp?4simple_rnn_5/simple_rnn_cell_9/MatMul/ReadVariableOp?6simple_rnn_5/simple_rnn_cell_9/MatMul_1/ReadVariableOp?simple_rnn_5/while^
simple_rnn_4/ShapeShapeinputs*
T0*
_output_shapes
:2
simple_rnn_4/Shape?
 simple_rnn_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 simple_rnn_4/strided_slice/stack?
"simple_rnn_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"simple_rnn_4/strided_slice/stack_1?
"simple_rnn_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"simple_rnn_4/strided_slice/stack_2?
simple_rnn_4/strided_sliceStridedSlicesimple_rnn_4/Shape:output:0)simple_rnn_4/strided_slice/stack:output:0+simple_rnn_4/strided_slice/stack_1:output:0+simple_rnn_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
simple_rnn_4/strided_slicew
simple_rnn_4/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
simple_rnn_4/zeros/mul/y?
simple_rnn_4/zeros/mulMul#simple_rnn_4/strided_slice:output:0!simple_rnn_4/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
simple_rnn_4/zeros/muly
simple_rnn_4/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
simple_rnn_4/zeros/Less/y?
simple_rnn_4/zeros/LessLesssimple_rnn_4/zeros/mul:z:0"simple_rnn_4/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
simple_rnn_4/zeros/Less}
simple_rnn_4/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
simple_rnn_4/zeros/packed/1?
simple_rnn_4/zeros/packedPack#simple_rnn_4/strided_slice:output:0$simple_rnn_4/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
simple_rnn_4/zeros/packedy
simple_rnn_4/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
simple_rnn_4/zeros/Const?
simple_rnn_4/zerosFill"simple_rnn_4/zeros/packed:output:0!simple_rnn_4/zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
simple_rnn_4/zeros?
simple_rnn_4/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
simple_rnn_4/transpose/perm?
simple_rnn_4/transpose	Transposeinputs$simple_rnn_4/transpose/perm:output:0*
T0*+
_output_shapes
:H?????????2
simple_rnn_4/transposev
simple_rnn_4/Shape_1Shapesimple_rnn_4/transpose:y:0*
T0*
_output_shapes
:2
simple_rnn_4/Shape_1?
"simple_rnn_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"simple_rnn_4/strided_slice_1/stack?
$simple_rnn_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2&
$simple_rnn_4/strided_slice_1/stack_1?
$simple_rnn_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$simple_rnn_4/strided_slice_1/stack_2?
simple_rnn_4/strided_slice_1StridedSlicesimple_rnn_4/Shape_1:output:0+simple_rnn_4/strided_slice_1/stack:output:0-simple_rnn_4/strided_slice_1/stack_1:output:0-simple_rnn_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
simple_rnn_4/strided_slice_1?
(simple_rnn_4/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2*
(simple_rnn_4/TensorArrayV2/element_shape?
simple_rnn_4/TensorArrayV2TensorListReserve1simple_rnn_4/TensorArrayV2/element_shape:output:0%simple_rnn_4/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
simple_rnn_4/TensorArrayV2?
Bsimple_rnn_4/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2D
Bsimple_rnn_4/TensorArrayUnstack/TensorListFromTensor/element_shape?
4simple_rnn_4/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorsimple_rnn_4/transpose:y:0Ksimple_rnn_4/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type026
4simple_rnn_4/TensorArrayUnstack/TensorListFromTensor?
"simple_rnn_4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"simple_rnn_4/strided_slice_2/stack?
$simple_rnn_4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2&
$simple_rnn_4/strided_slice_2/stack_1?
$simple_rnn_4/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$simple_rnn_4/strided_slice_2/stack_2?
simple_rnn_4/strided_slice_2StridedSlicesimple_rnn_4/transpose:y:0+simple_rnn_4/strided_slice_2/stack:output:0-simple_rnn_4/strided_slice_2/stack_1:output:0-simple_rnn_4/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
simple_rnn_4/strided_slice_2?
4simple_rnn_4/simple_rnn_cell_8/MatMul/ReadVariableOpReadVariableOp=simple_rnn_4_simple_rnn_cell_8_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype026
4simple_rnn_4/simple_rnn_cell_8/MatMul/ReadVariableOp?
%simple_rnn_4/simple_rnn_cell_8/MatMul	MLCMatMul%simple_rnn_4/strided_slice_2:output:0<simple_rnn_4/simple_rnn_cell_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2'
%simple_rnn_4/simple_rnn_cell_8/MatMul?
5simple_rnn_4/simple_rnn_cell_8/BiasAdd/ReadVariableOpReadVariableOp>simple_rnn_4_simple_rnn_cell_8_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype027
5simple_rnn_4/simple_rnn_cell_8/BiasAdd/ReadVariableOp?
&simple_rnn_4/simple_rnn_cell_8/BiasAddBiasAdd/simple_rnn_4/simple_rnn_cell_8/MatMul:product:0=simple_rnn_4/simple_rnn_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2(
&simple_rnn_4/simple_rnn_cell_8/BiasAdd?
6simple_rnn_4/simple_rnn_cell_8/MatMul_1/ReadVariableOpReadVariableOp?simple_rnn_4_simple_rnn_cell_8_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype028
6simple_rnn_4/simple_rnn_cell_8/MatMul_1/ReadVariableOp?
'simple_rnn_4/simple_rnn_cell_8/MatMul_1	MLCMatMulsimple_rnn_4/zeros:output:0>simple_rnn_4/simple_rnn_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2)
'simple_rnn_4/simple_rnn_cell_8/MatMul_1?
"simple_rnn_4/simple_rnn_cell_8/addAddV2/simple_rnn_4/simple_rnn_cell_8/BiasAdd:output:01simple_rnn_4/simple_rnn_cell_8/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2$
"simple_rnn_4/simple_rnn_cell_8/add?
#simple_rnn_4/simple_rnn_cell_8/TanhTanh&simple_rnn_4/simple_rnn_cell_8/add:z:0*
T0*(
_output_shapes
:??????????2%
#simple_rnn_4/simple_rnn_cell_8/Tanh?
*simple_rnn_4/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2,
*simple_rnn_4/TensorArrayV2_1/element_shape?
simple_rnn_4/TensorArrayV2_1TensorListReserve3simple_rnn_4/TensorArrayV2_1/element_shape:output:0%simple_rnn_4/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
simple_rnn_4/TensorArrayV2_1h
simple_rnn_4/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
simple_rnn_4/time?
%simple_rnn_4/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2'
%simple_rnn_4/while/maximum_iterations?
simple_rnn_4/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2!
simple_rnn_4/while/loop_counter?
simple_rnn_4/whileWhile(simple_rnn_4/while/loop_counter:output:0.simple_rnn_4/while/maximum_iterations:output:0simple_rnn_4/time:output:0%simple_rnn_4/TensorArrayV2_1:handle:0simple_rnn_4/zeros:output:0%simple_rnn_4/strided_slice_1:output:0Dsimple_rnn_4/TensorArrayUnstack/TensorListFromTensor:output_handle:0=simple_rnn_4_simple_rnn_cell_8_matmul_readvariableop_resource>simple_rnn_4_simple_rnn_cell_8_biasadd_readvariableop_resource?simple_rnn_4_simple_rnn_cell_8_matmul_1_readvariableop_resource*
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
simple_rnn_4_while_body_100666**
cond"R 
simple_rnn_4_while_cond_100665*9
output_shapes(
&: : : : :??????????: : : : : *
parallel_iterations 2
simple_rnn_4/while?
=simple_rnn_4/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2?
=simple_rnn_4/TensorArrayV2Stack/TensorListStack/element_shape?
/simple_rnn_4/TensorArrayV2Stack/TensorListStackTensorListStacksimple_rnn_4/while:output:3Fsimple_rnn_4/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:H??????????*
element_dtype021
/simple_rnn_4/TensorArrayV2Stack/TensorListStack?
"simple_rnn_4/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2$
"simple_rnn_4/strided_slice_3/stack?
$simple_rnn_4/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2&
$simple_rnn_4/strided_slice_3/stack_1?
$simple_rnn_4/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$simple_rnn_4/strided_slice_3/stack_2?
simple_rnn_4/strided_slice_3StridedSlice8simple_rnn_4/TensorArrayV2Stack/TensorListStack:tensor:0+simple_rnn_4/strided_slice_3/stack:output:0-simple_rnn_4/strided_slice_3/stack_1:output:0-simple_rnn_4/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
simple_rnn_4/strided_slice_3?
simple_rnn_4/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
simple_rnn_4/transpose_1/perm?
simple_rnn_4/transpose_1	Transpose8simple_rnn_4/TensorArrayV2Stack/TensorListStack:tensor:0&simple_rnn_4/transpose_1/perm:output:0*
T0*,
_output_shapes
:?????????H?2
simple_rnn_4/transpose_1?
activation_10/ReluRelusimple_rnn_4/transpose_1:y:0*
T0*,
_output_shapes
:?????????H?2
activation_10/Reluw
dropout_4/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout_4/dropout/Const?
dropout_4/dropout/MulMul activation_10/Relu:activations:0 dropout_4/dropout/Const:output:0*
T0*,
_output_shapes
:?????????H?2
dropout_4/dropout/Mulu
dropout_4/dropout/rateConst*
_output_shapes
: *
dtype0*
valueB
 *???=2
dropout_4/dropout/rater
dropout_4/dropout/seedConst*
_output_shapes
: *
dtype0*
value	B : 2
dropout_4/dropout/seed?
dropout_4/dropout
MLCDropout activation_10/Relu:activations:0dropout_4/dropout/rate:output:0dropout_4/dropout/seed:output:0*
T0*,
_output_shapes
:?????????H?2
dropout_4/dropoutr
simple_rnn_5/ShapeShapedropout_4/dropout:output:0*
T0*
_output_shapes
:2
simple_rnn_5/Shape?
 simple_rnn_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 simple_rnn_5/strided_slice/stack?
"simple_rnn_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"simple_rnn_5/strided_slice/stack_1?
"simple_rnn_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"simple_rnn_5/strided_slice/stack_2?
simple_rnn_5/strided_sliceStridedSlicesimple_rnn_5/Shape:output:0)simple_rnn_5/strided_slice/stack:output:0+simple_rnn_5/strided_slice/stack_1:output:0+simple_rnn_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
simple_rnn_5/strided_slicev
simple_rnn_5/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
simple_rnn_5/zeros/mul/y?
simple_rnn_5/zeros/mulMul#simple_rnn_5/strided_slice:output:0!simple_rnn_5/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
simple_rnn_5/zeros/muly
simple_rnn_5/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
simple_rnn_5/zeros/Less/y?
simple_rnn_5/zeros/LessLesssimple_rnn_5/zeros/mul:z:0"simple_rnn_5/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
simple_rnn_5/zeros/Less|
simple_rnn_5/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
simple_rnn_5/zeros/packed/1?
simple_rnn_5/zeros/packedPack#simple_rnn_5/strided_slice:output:0$simple_rnn_5/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
simple_rnn_5/zeros/packedy
simple_rnn_5/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
simple_rnn_5/zeros/Const?
simple_rnn_5/zerosFill"simple_rnn_5/zeros/packed:output:0!simple_rnn_5/zeros/Const:output:0*
T0*'
_output_shapes
:?????????22
simple_rnn_5/zeros?
simple_rnn_5/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
simple_rnn_5/transpose/perm?
simple_rnn_5/transpose	Transposedropout_4/dropout:output:0$simple_rnn_5/transpose/perm:output:0*
T0*,
_output_shapes
:H??????????2
simple_rnn_5/transposev
simple_rnn_5/Shape_1Shapesimple_rnn_5/transpose:y:0*
T0*
_output_shapes
:2
simple_rnn_5/Shape_1?
"simple_rnn_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"simple_rnn_5/strided_slice_1/stack?
$simple_rnn_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2&
$simple_rnn_5/strided_slice_1/stack_1?
$simple_rnn_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$simple_rnn_5/strided_slice_1/stack_2?
simple_rnn_5/strided_slice_1StridedSlicesimple_rnn_5/Shape_1:output:0+simple_rnn_5/strided_slice_1/stack:output:0-simple_rnn_5/strided_slice_1/stack_1:output:0-simple_rnn_5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
simple_rnn_5/strided_slice_1?
(simple_rnn_5/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2*
(simple_rnn_5/TensorArrayV2/element_shape?
simple_rnn_5/TensorArrayV2TensorListReserve1simple_rnn_5/TensorArrayV2/element_shape:output:0%simple_rnn_5/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
simple_rnn_5/TensorArrayV2?
Bsimple_rnn_5/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2D
Bsimple_rnn_5/TensorArrayUnstack/TensorListFromTensor/element_shape?
4simple_rnn_5/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorsimple_rnn_5/transpose:y:0Ksimple_rnn_5/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type026
4simple_rnn_5/TensorArrayUnstack/TensorListFromTensor?
"simple_rnn_5/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"simple_rnn_5/strided_slice_2/stack?
$simple_rnn_5/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2&
$simple_rnn_5/strided_slice_2/stack_1?
$simple_rnn_5/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$simple_rnn_5/strided_slice_2/stack_2?
simple_rnn_5/strided_slice_2StridedSlicesimple_rnn_5/transpose:y:0+simple_rnn_5/strided_slice_2/stack:output:0-simple_rnn_5/strided_slice_2/stack_1:output:0-simple_rnn_5/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
simple_rnn_5/strided_slice_2?
4simple_rnn_5/simple_rnn_cell_9/MatMul/ReadVariableOpReadVariableOp=simple_rnn_5_simple_rnn_cell_9_matmul_readvariableop_resource*
_output_shapes
:	?2*
dtype026
4simple_rnn_5/simple_rnn_cell_9/MatMul/ReadVariableOp?
%simple_rnn_5/simple_rnn_cell_9/MatMul	MLCMatMul%simple_rnn_5/strided_slice_2:output:0<simple_rnn_5/simple_rnn_cell_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22'
%simple_rnn_5/simple_rnn_cell_9/MatMul?
5simple_rnn_5/simple_rnn_cell_9/BiasAdd/ReadVariableOpReadVariableOp>simple_rnn_5_simple_rnn_cell_9_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype027
5simple_rnn_5/simple_rnn_cell_9/BiasAdd/ReadVariableOp?
&simple_rnn_5/simple_rnn_cell_9/BiasAddBiasAdd/simple_rnn_5/simple_rnn_cell_9/MatMul:product:0=simple_rnn_5/simple_rnn_cell_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22(
&simple_rnn_5/simple_rnn_cell_9/BiasAdd?
6simple_rnn_5/simple_rnn_cell_9/MatMul_1/ReadVariableOpReadVariableOp?simple_rnn_5_simple_rnn_cell_9_matmul_1_readvariableop_resource*
_output_shapes

:22*
dtype028
6simple_rnn_5/simple_rnn_cell_9/MatMul_1/ReadVariableOp?
'simple_rnn_5/simple_rnn_cell_9/MatMul_1	MLCMatMulsimple_rnn_5/zeros:output:0>simple_rnn_5/simple_rnn_cell_9/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22)
'simple_rnn_5/simple_rnn_cell_9/MatMul_1?
"simple_rnn_5/simple_rnn_cell_9/addAddV2/simple_rnn_5/simple_rnn_cell_9/BiasAdd:output:01simple_rnn_5/simple_rnn_cell_9/MatMul_1:product:0*
T0*'
_output_shapes
:?????????22$
"simple_rnn_5/simple_rnn_cell_9/add?
#simple_rnn_5/simple_rnn_cell_9/TanhTanh&simple_rnn_5/simple_rnn_cell_9/add:z:0*
T0*'
_output_shapes
:?????????22%
#simple_rnn_5/simple_rnn_cell_9/Tanh?
*simple_rnn_5/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   2,
*simple_rnn_5/TensorArrayV2_1/element_shape?
simple_rnn_5/TensorArrayV2_1TensorListReserve3simple_rnn_5/TensorArrayV2_1/element_shape:output:0%simple_rnn_5/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
simple_rnn_5/TensorArrayV2_1h
simple_rnn_5/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
simple_rnn_5/time?
%simple_rnn_5/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2'
%simple_rnn_5/while/maximum_iterations?
simple_rnn_5/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2!
simple_rnn_5/while/loop_counter?
simple_rnn_5/whileWhile(simple_rnn_5/while/loop_counter:output:0.simple_rnn_5/while/maximum_iterations:output:0simple_rnn_5/time:output:0%simple_rnn_5/TensorArrayV2_1:handle:0simple_rnn_5/zeros:output:0%simple_rnn_5/strided_slice_1:output:0Dsimple_rnn_5/TensorArrayUnstack/TensorListFromTensor:output_handle:0=simple_rnn_5_simple_rnn_cell_9_matmul_readvariableop_resource>simple_rnn_5_simple_rnn_cell_9_biasadd_readvariableop_resource?simple_rnn_5_simple_rnn_cell_9_matmul_1_readvariableop_resource*
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
simple_rnn_5_while_body_100780**
cond"R 
simple_rnn_5_while_cond_100779*8
output_shapes'
%: : : : :?????????2: : : : : *
parallel_iterations 2
simple_rnn_5/while?
=simple_rnn_5/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   2?
=simple_rnn_5/TensorArrayV2Stack/TensorListStack/element_shape?
/simple_rnn_5/TensorArrayV2Stack/TensorListStackTensorListStacksimple_rnn_5/while:output:3Fsimple_rnn_5/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:H?????????2*
element_dtype021
/simple_rnn_5/TensorArrayV2Stack/TensorListStack?
"simple_rnn_5/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2$
"simple_rnn_5/strided_slice_3/stack?
$simple_rnn_5/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2&
$simple_rnn_5/strided_slice_3/stack_1?
$simple_rnn_5/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$simple_rnn_5/strided_slice_3/stack_2?
simple_rnn_5/strided_slice_3StridedSlice8simple_rnn_5/TensorArrayV2Stack/TensorListStack:tensor:0+simple_rnn_5/strided_slice_3/stack:output:0-simple_rnn_5/strided_slice_3/stack_1:output:0-simple_rnn_5/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????2*
shrink_axis_mask2
simple_rnn_5/strided_slice_3?
simple_rnn_5/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
simple_rnn_5/transpose_1/perm?
simple_rnn_5/transpose_1	Transpose8simple_rnn_5/TensorArrayV2Stack/TensorListStack:tensor:0&simple_rnn_5/transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????H22
simple_rnn_5/transpose_1?
activation_11/ReluRelu%simple_rnn_5/strided_slice_3:output:0*
T0*'
_output_shapes
:?????????22
activation_11/Reluw
dropout_5/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout_5/dropout/Const?
dropout_5/dropout/MulMul activation_11/Relu:activations:0 dropout_5/dropout/Const:output:0*
T0*'
_output_shapes
:?????????22
dropout_5/dropout/Mulu
dropout_5/dropout/rateConst*
_output_shapes
: *
dtype0*
valueB
 *???=2
dropout_5/dropout/rater
dropout_5/dropout/seedConst*
_output_shapes
: *
dtype0*
value	B : 2
dropout_5/dropout/seed?
dropout_5/dropout
MLCDropout activation_11/Relu:activations:0dropout_5/dropout/rate:output:0dropout_5/dropout/seed:output:0*
T0*'
_output_shapes
:?????????22
dropout_5/dropout?
!dense_14/MLCMatMul/ReadVariableOpReadVariableOp*dense_14_mlcmatmul_readvariableop_resource*
_output_shapes

:2$*
dtype02#
!dense_14/MLCMatMul/ReadVariableOp?
dense_14/MLCMatMul	MLCMatMuldropout_5/dropout:output:0)dense_14/MLCMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????$2
dense_14/MLCMatMul?
dense_14/BiasAdd/ReadVariableOpReadVariableOp(dense_14_biasadd_readvariableop_resource*
_output_shapes
:$*
dtype02!
dense_14/BiasAdd/ReadVariableOp?
dense_14/BiasAddBiasAdddense_14/MLCMatMul:product:0'dense_14/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????$2
dense_14/BiasAdds
dense_14/TanhTanhdense_14/BiasAdd:output:0*
T0*'
_output_shapes
:?????????$2
dense_14/Tanh?
IdentityIdentitydense_14/Tanh:y:0 ^dense_14/BiasAdd/ReadVariableOp"^dense_14/MLCMatMul/ReadVariableOp6^simple_rnn_4/simple_rnn_cell_8/BiasAdd/ReadVariableOp5^simple_rnn_4/simple_rnn_cell_8/MatMul/ReadVariableOp7^simple_rnn_4/simple_rnn_cell_8/MatMul_1/ReadVariableOp^simple_rnn_4/while6^simple_rnn_5/simple_rnn_cell_9/BiasAdd/ReadVariableOp5^simple_rnn_5/simple_rnn_cell_9/MatMul/ReadVariableOp7^simple_rnn_5/simple_rnn_cell_9/MatMul_1/ReadVariableOp^simple_rnn_5/while*
T0*'
_output_shapes
:?????????$2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:?????????H::::::::2B
dense_14/BiasAdd/ReadVariableOpdense_14/BiasAdd/ReadVariableOp2F
!dense_14/MLCMatMul/ReadVariableOp!dense_14/MLCMatMul/ReadVariableOp2n
5simple_rnn_4/simple_rnn_cell_8/BiasAdd/ReadVariableOp5simple_rnn_4/simple_rnn_cell_8/BiasAdd/ReadVariableOp2l
4simple_rnn_4/simple_rnn_cell_8/MatMul/ReadVariableOp4simple_rnn_4/simple_rnn_cell_8/MatMul/ReadVariableOp2p
6simple_rnn_4/simple_rnn_cell_8/MatMul_1/ReadVariableOp6simple_rnn_4/simple_rnn_cell_8/MatMul_1/ReadVariableOp2(
simple_rnn_4/whilesimple_rnn_4/while2n
5simple_rnn_5/simple_rnn_cell_9/BiasAdd/ReadVariableOp5simple_rnn_5/simple_rnn_cell_9/BiasAdd/ReadVariableOp2l
4simple_rnn_5/simple_rnn_cell_9/MatMul/ReadVariableOp4simple_rnn_5/simple_rnn_cell_9/MatMul/ReadVariableOp2p
6simple_rnn_5/simple_rnn_cell_9/MatMul_1/ReadVariableOp6simple_rnn_5/simple_rnn_cell_9/MatMul_1/ReadVariableOp2(
simple_rnn_5/whilesimple_rnn_5/while:S O
+
_output_shapes
:?????????H
 
_user_specified_nameinputs
?3
?
while_body_101816
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0<
8while_simple_rnn_cell_9_matmul_readvariableop_resource_0=
9while_simple_rnn_cell_9_biasadd_readvariableop_resource_0>
:while_simple_rnn_cell_9_matmul_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor:
6while_simple_rnn_cell_9_matmul_readvariableop_resource;
7while_simple_rnn_cell_9_biasadd_readvariableop_resource<
8while_simple_rnn_cell_9_matmul_1_readvariableop_resource??.while/simple_rnn_cell_9/BiasAdd/ReadVariableOp?-while/simple_rnn_cell_9/MatMul/ReadVariableOp?/while/simple_rnn_cell_9/MatMul_1/ReadVariableOp?
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
-while/simple_rnn_cell_9/MatMul/ReadVariableOpReadVariableOp8while_simple_rnn_cell_9_matmul_readvariableop_resource_0*
_output_shapes
:	?2*
dtype02/
-while/simple_rnn_cell_9/MatMul/ReadVariableOp?
while/simple_rnn_cell_9/MatMul	MLCMatMul0while/TensorArrayV2Read/TensorListGetItem:item:05while/simple_rnn_cell_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22 
while/simple_rnn_cell_9/MatMul?
.while/simple_rnn_cell_9/BiasAdd/ReadVariableOpReadVariableOp9while_simple_rnn_cell_9_biasadd_readvariableop_resource_0*
_output_shapes
:2*
dtype020
.while/simple_rnn_cell_9/BiasAdd/ReadVariableOp?
while/simple_rnn_cell_9/BiasAddBiasAdd(while/simple_rnn_cell_9/MatMul:product:06while/simple_rnn_cell_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22!
while/simple_rnn_cell_9/BiasAdd?
/while/simple_rnn_cell_9/MatMul_1/ReadVariableOpReadVariableOp:while_simple_rnn_cell_9_matmul_1_readvariableop_resource_0*
_output_shapes

:22*
dtype021
/while/simple_rnn_cell_9/MatMul_1/ReadVariableOp?
 while/simple_rnn_cell_9/MatMul_1	MLCMatMulwhile_placeholder_27while/simple_rnn_cell_9/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22"
 while/simple_rnn_cell_9/MatMul_1?
while/simple_rnn_cell_9/addAddV2(while/simple_rnn_cell_9/BiasAdd:output:0*while/simple_rnn_cell_9/MatMul_1:product:0*
T0*'
_output_shapes
:?????????22
while/simple_rnn_cell_9/add?
while/simple_rnn_cell_9/TanhTanhwhile/simple_rnn_cell_9/add:z:0*
T0*'
_output_shapes
:?????????22
while/simple_rnn_cell_9/Tanh?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder while/simple_rnn_cell_9/Tanh:y:0*
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
while/IdentityIdentitywhile/add_1:z:0/^while/simple_rnn_cell_9/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_9/MatMul/ReadVariableOp0^while/simple_rnn_cell_9/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations/^while/simple_rnn_cell_9/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_9/MatMul/ReadVariableOp0^while/simple_rnn_cell_9/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0/^while/simple_rnn_cell_9/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_9/MatMul/ReadVariableOp0^while/simple_rnn_cell_9/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0/^while/simple_rnn_cell_9/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_9/MatMul/ReadVariableOp0^while/simple_rnn_cell_9/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity while/simple_rnn_cell_9/Tanh:y:0/^while/simple_rnn_cell_9/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_9/MatMul/ReadVariableOp0^while/simple_rnn_cell_9/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????22
while/Identity_4")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"t
7while_simple_rnn_cell_9_biasadd_readvariableop_resource9while_simple_rnn_cell_9_biasadd_readvariableop_resource_0"v
8while_simple_rnn_cell_9_matmul_1_readvariableop_resource:while_simple_rnn_cell_9_matmul_1_readvariableop_resource_0"r
6while_simple_rnn_cell_9_matmul_readvariableop_resource8while_simple_rnn_cell_9_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :?????????2: : :::2`
.while/simple_rnn_cell_9/BiasAdd/ReadVariableOp.while/simple_rnn_cell_9/BiasAdd/ReadVariableOp2^
-while/simple_rnn_cell_9/MatMul/ReadVariableOp-while/simple_rnn_cell_9/MatMul/ReadVariableOp2b
/while/simple_rnn_cell_9/MatMul_1/ReadVariableOp/while/simple_rnn_cell_9/MatMul_1/ReadVariableOp: 
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
?G
?
H__inference_simple_rnn_5_layer_call_and_return_conditional_losses_100255

inputs4
0simple_rnn_cell_9_matmul_readvariableop_resource5
1simple_rnn_cell_9_biasadd_readvariableop_resource6
2simple_rnn_cell_9_matmul_1_readvariableop_resource
identity??(simple_rnn_cell_9/BiasAdd/ReadVariableOp?'simple_rnn_cell_9/MatMul/ReadVariableOp?)simple_rnn_cell_9/MatMul_1/ReadVariableOp?whileD
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
:H??????????2
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
'simple_rnn_cell_9/MatMul/ReadVariableOpReadVariableOp0simple_rnn_cell_9_matmul_readvariableop_resource*
_output_shapes
:	?2*
dtype02)
'simple_rnn_cell_9/MatMul/ReadVariableOp?
simple_rnn_cell_9/MatMul	MLCMatMulstrided_slice_2:output:0/simple_rnn_cell_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
simple_rnn_cell_9/MatMul?
(simple_rnn_cell_9/BiasAdd/ReadVariableOpReadVariableOp1simple_rnn_cell_9_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02*
(simple_rnn_cell_9/BiasAdd/ReadVariableOp?
simple_rnn_cell_9/BiasAddBiasAdd"simple_rnn_cell_9/MatMul:product:00simple_rnn_cell_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
simple_rnn_cell_9/BiasAdd?
)simple_rnn_cell_9/MatMul_1/ReadVariableOpReadVariableOp2simple_rnn_cell_9_matmul_1_readvariableop_resource*
_output_shapes

:22*
dtype02+
)simple_rnn_cell_9/MatMul_1/ReadVariableOp?
simple_rnn_cell_9/MatMul_1	MLCMatMulzeros:output:01simple_rnn_cell_9/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
simple_rnn_cell_9/MatMul_1?
simple_rnn_cell_9/addAddV2"simple_rnn_cell_9/BiasAdd:output:0$simple_rnn_cell_9/MatMul_1:product:0*
T0*'
_output_shapes
:?????????22
simple_rnn_cell_9/add?
simple_rnn_cell_9/TanhTanhsimple_rnn_cell_9/add:z:0*
T0*'
_output_shapes
:?????????22
simple_rnn_cell_9/Tanh?
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:00simple_rnn_cell_9_matmul_readvariableop_resource1simple_rnn_cell_9_biasadd_readvariableop_resource2simple_rnn_cell_9_matmul_1_readvariableop_resource*
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
while_body_100189*
condR
while_cond_100188*8
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
:H?????????2*
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
:?????????H22
transpose_1?
IdentityIdentitystrided_slice_3:output:0)^simple_rnn_cell_9/BiasAdd/ReadVariableOp(^simple_rnn_cell_9/MatMul/ReadVariableOp*^simple_rnn_cell_9/MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????H?:::2T
(simple_rnn_cell_9/BiasAdd/ReadVariableOp(simple_rnn_cell_9/BiasAdd/ReadVariableOp2R
'simple_rnn_cell_9/MatMul/ReadVariableOp'simple_rnn_cell_9/MatMul/ReadVariableOp2V
)simple_rnn_cell_9/MatMul_1/ReadVariableOp)simple_rnn_cell_9/MatMul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:?????????H?
 
_user_specified_nameinputs
?
?
+sequential_12_simple_rnn_4_while_cond_98637R
Nsequential_12_simple_rnn_4_while_sequential_12_simple_rnn_4_while_loop_counterX
Tsequential_12_simple_rnn_4_while_sequential_12_simple_rnn_4_while_maximum_iterations0
,sequential_12_simple_rnn_4_while_placeholder2
.sequential_12_simple_rnn_4_while_placeholder_12
.sequential_12_simple_rnn_4_while_placeholder_2T
Psequential_12_simple_rnn_4_while_less_sequential_12_simple_rnn_4_strided_slice_1i
esequential_12_simple_rnn_4_while_sequential_12_simple_rnn_4_while_cond_98637___redundant_placeholder0i
esequential_12_simple_rnn_4_while_sequential_12_simple_rnn_4_while_cond_98637___redundant_placeholder1i
esequential_12_simple_rnn_4_while_sequential_12_simple_rnn_4_while_cond_98637___redundant_placeholder2i
esequential_12_simple_rnn_4_while_sequential_12_simple_rnn_4_while_cond_98637___redundant_placeholder3-
)sequential_12_simple_rnn_4_while_identity
?
%sequential_12/simple_rnn_4/while/LessLess,sequential_12_simple_rnn_4_while_placeholderPsequential_12_simple_rnn_4_while_less_sequential_12_simple_rnn_4_strided_slice_1*
T0*
_output_shapes
: 2'
%sequential_12/simple_rnn_4/while/Less?
)sequential_12/simple_rnn_4/while/IdentityIdentity)sequential_12/simple_rnn_4/while/Less:z:0*
T0
*
_output_shapes
: 2+
)sequential_12/simple_rnn_4/while/Identity"_
)sequential_12_simple_rnn_4_while_identity2sequential_12/simple_rnn_4/while/Identity:output:0*A
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
?
I__inference_sequential_12_layer_call_and_return_conditional_losses_100492
simple_rnn_4_input
simple_rnn_4_100468
simple_rnn_4_100470
simple_rnn_4_100472
simple_rnn_5_100477
simple_rnn_5_100479
simple_rnn_5_100481
dense_14_100486
dense_14_100488
identity?? dense_14/StatefulPartitionedCall?$simple_rnn_4/StatefulPartitionedCall?$simple_rnn_5/StatefulPartitionedCall?
$simple_rnn_4/StatefulPartitionedCallStatefulPartitionedCallsimple_rnn_4_inputsimple_rnn_4_100468simple_rnn_4_100470simple_rnn_4_100472*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????H?*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_simple_rnn_4_layer_call_and_return_conditional_losses_1000742&
$simple_rnn_4/StatefulPartitionedCall?
activation_10/PartitionedCallPartitionedCall-simple_rnn_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????H?* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_activation_10_layer_call_and_return_conditional_losses_1001092
activation_10/PartitionedCall?
dropout_4/PartitionedCallPartitionedCall&activation_10/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????H?* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_4_layer_call_and_return_conditional_losses_1001312
dropout_4/PartitionedCall?
$simple_rnn_5/StatefulPartitionedCallStatefulPartitionedCall"dropout_4/PartitionedCall:output:0simple_rnn_5_100477simple_rnn_5_100479simple_rnn_5_100481*
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
H__inference_simple_rnn_5_layer_call_and_return_conditional_losses_1003672&
$simple_rnn_5/StatefulPartitionedCall?
activation_11/PartitionedCallPartitionedCall-simple_rnn_5/StatefulPartitionedCall:output:0*
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
I__inference_activation_11_layer_call_and_return_conditional_losses_1004022
activation_11/PartitionedCall?
dropout_5/PartitionedCallPartitionedCall&activation_11/PartitionedCall:output:0*
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
E__inference_dropout_5_layer_call_and_return_conditional_losses_1004242
dropout_5/PartitionedCall?
 dense_14/StatefulPartitionedCallStatefulPartitionedCall"dropout_5/PartitionedCall:output:0dense_14_100486dense_14_100488*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????$*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_14_layer_call_and_return_conditional_losses_1004482"
 dense_14/StatefulPartitionedCall?
IdentityIdentity)dense_14/StatefulPartitionedCall:output:0!^dense_14/StatefulPartitionedCall%^simple_rnn_4/StatefulPartitionedCall%^simple_rnn_5/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????$2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:?????????H::::::::2D
 dense_14/StatefulPartitionedCall dense_14/StatefulPartitionedCall2L
$simple_rnn_4/StatefulPartitionedCall$simple_rnn_4/StatefulPartitionedCall2L
$simple_rnn_5/StatefulPartitionedCall$simple_rnn_5/StatefulPartitionedCall:_ [
+
_output_shapes
:?????????H
,
_user_specified_namesimple_rnn_4_input
??
?
I__inference_sequential_12_layer_call_and_return_conditional_losses_101090

inputsA
=simple_rnn_4_simple_rnn_cell_8_matmul_readvariableop_resourceB
>simple_rnn_4_simple_rnn_cell_8_biasadd_readvariableop_resourceC
?simple_rnn_4_simple_rnn_cell_8_matmul_1_readvariableop_resourceA
=simple_rnn_5_simple_rnn_cell_9_matmul_readvariableop_resourceB
>simple_rnn_5_simple_rnn_cell_9_biasadd_readvariableop_resourceC
?simple_rnn_5_simple_rnn_cell_9_matmul_1_readvariableop_resource.
*dense_14_mlcmatmul_readvariableop_resource,
(dense_14_biasadd_readvariableop_resource
identity??dense_14/BiasAdd/ReadVariableOp?!dense_14/MLCMatMul/ReadVariableOp?5simple_rnn_4/simple_rnn_cell_8/BiasAdd/ReadVariableOp?4simple_rnn_4/simple_rnn_cell_8/MatMul/ReadVariableOp?6simple_rnn_4/simple_rnn_cell_8/MatMul_1/ReadVariableOp?simple_rnn_4/while?5simple_rnn_5/simple_rnn_cell_9/BiasAdd/ReadVariableOp?4simple_rnn_5/simple_rnn_cell_9/MatMul/ReadVariableOp?6simple_rnn_5/simple_rnn_cell_9/MatMul_1/ReadVariableOp?simple_rnn_5/while^
simple_rnn_4/ShapeShapeinputs*
T0*
_output_shapes
:2
simple_rnn_4/Shape?
 simple_rnn_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 simple_rnn_4/strided_slice/stack?
"simple_rnn_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"simple_rnn_4/strided_slice/stack_1?
"simple_rnn_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"simple_rnn_4/strided_slice/stack_2?
simple_rnn_4/strided_sliceStridedSlicesimple_rnn_4/Shape:output:0)simple_rnn_4/strided_slice/stack:output:0+simple_rnn_4/strided_slice/stack_1:output:0+simple_rnn_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
simple_rnn_4/strided_slicew
simple_rnn_4/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
simple_rnn_4/zeros/mul/y?
simple_rnn_4/zeros/mulMul#simple_rnn_4/strided_slice:output:0!simple_rnn_4/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
simple_rnn_4/zeros/muly
simple_rnn_4/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
simple_rnn_4/zeros/Less/y?
simple_rnn_4/zeros/LessLesssimple_rnn_4/zeros/mul:z:0"simple_rnn_4/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
simple_rnn_4/zeros/Less}
simple_rnn_4/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
simple_rnn_4/zeros/packed/1?
simple_rnn_4/zeros/packedPack#simple_rnn_4/strided_slice:output:0$simple_rnn_4/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
simple_rnn_4/zeros/packedy
simple_rnn_4/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
simple_rnn_4/zeros/Const?
simple_rnn_4/zerosFill"simple_rnn_4/zeros/packed:output:0!simple_rnn_4/zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
simple_rnn_4/zeros?
simple_rnn_4/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
simple_rnn_4/transpose/perm?
simple_rnn_4/transpose	Transposeinputs$simple_rnn_4/transpose/perm:output:0*
T0*+
_output_shapes
:H?????????2
simple_rnn_4/transposev
simple_rnn_4/Shape_1Shapesimple_rnn_4/transpose:y:0*
T0*
_output_shapes
:2
simple_rnn_4/Shape_1?
"simple_rnn_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"simple_rnn_4/strided_slice_1/stack?
$simple_rnn_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2&
$simple_rnn_4/strided_slice_1/stack_1?
$simple_rnn_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$simple_rnn_4/strided_slice_1/stack_2?
simple_rnn_4/strided_slice_1StridedSlicesimple_rnn_4/Shape_1:output:0+simple_rnn_4/strided_slice_1/stack:output:0-simple_rnn_4/strided_slice_1/stack_1:output:0-simple_rnn_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
simple_rnn_4/strided_slice_1?
(simple_rnn_4/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2*
(simple_rnn_4/TensorArrayV2/element_shape?
simple_rnn_4/TensorArrayV2TensorListReserve1simple_rnn_4/TensorArrayV2/element_shape:output:0%simple_rnn_4/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
simple_rnn_4/TensorArrayV2?
Bsimple_rnn_4/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2D
Bsimple_rnn_4/TensorArrayUnstack/TensorListFromTensor/element_shape?
4simple_rnn_4/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorsimple_rnn_4/transpose:y:0Ksimple_rnn_4/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type026
4simple_rnn_4/TensorArrayUnstack/TensorListFromTensor?
"simple_rnn_4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"simple_rnn_4/strided_slice_2/stack?
$simple_rnn_4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2&
$simple_rnn_4/strided_slice_2/stack_1?
$simple_rnn_4/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$simple_rnn_4/strided_slice_2/stack_2?
simple_rnn_4/strided_slice_2StridedSlicesimple_rnn_4/transpose:y:0+simple_rnn_4/strided_slice_2/stack:output:0-simple_rnn_4/strided_slice_2/stack_1:output:0-simple_rnn_4/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
simple_rnn_4/strided_slice_2?
4simple_rnn_4/simple_rnn_cell_8/MatMul/ReadVariableOpReadVariableOp=simple_rnn_4_simple_rnn_cell_8_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype026
4simple_rnn_4/simple_rnn_cell_8/MatMul/ReadVariableOp?
%simple_rnn_4/simple_rnn_cell_8/MatMul	MLCMatMul%simple_rnn_4/strided_slice_2:output:0<simple_rnn_4/simple_rnn_cell_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2'
%simple_rnn_4/simple_rnn_cell_8/MatMul?
5simple_rnn_4/simple_rnn_cell_8/BiasAdd/ReadVariableOpReadVariableOp>simple_rnn_4_simple_rnn_cell_8_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype027
5simple_rnn_4/simple_rnn_cell_8/BiasAdd/ReadVariableOp?
&simple_rnn_4/simple_rnn_cell_8/BiasAddBiasAdd/simple_rnn_4/simple_rnn_cell_8/MatMul:product:0=simple_rnn_4/simple_rnn_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2(
&simple_rnn_4/simple_rnn_cell_8/BiasAdd?
6simple_rnn_4/simple_rnn_cell_8/MatMul_1/ReadVariableOpReadVariableOp?simple_rnn_4_simple_rnn_cell_8_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype028
6simple_rnn_4/simple_rnn_cell_8/MatMul_1/ReadVariableOp?
'simple_rnn_4/simple_rnn_cell_8/MatMul_1	MLCMatMulsimple_rnn_4/zeros:output:0>simple_rnn_4/simple_rnn_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2)
'simple_rnn_4/simple_rnn_cell_8/MatMul_1?
"simple_rnn_4/simple_rnn_cell_8/addAddV2/simple_rnn_4/simple_rnn_cell_8/BiasAdd:output:01simple_rnn_4/simple_rnn_cell_8/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2$
"simple_rnn_4/simple_rnn_cell_8/add?
#simple_rnn_4/simple_rnn_cell_8/TanhTanh&simple_rnn_4/simple_rnn_cell_8/add:z:0*
T0*(
_output_shapes
:??????????2%
#simple_rnn_4/simple_rnn_cell_8/Tanh?
*simple_rnn_4/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2,
*simple_rnn_4/TensorArrayV2_1/element_shape?
simple_rnn_4/TensorArrayV2_1TensorListReserve3simple_rnn_4/TensorArrayV2_1/element_shape:output:0%simple_rnn_4/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
simple_rnn_4/TensorArrayV2_1h
simple_rnn_4/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
simple_rnn_4/time?
%simple_rnn_4/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2'
%simple_rnn_4/while/maximum_iterations?
simple_rnn_4/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2!
simple_rnn_4/while/loop_counter?
simple_rnn_4/whileWhile(simple_rnn_4/while/loop_counter:output:0.simple_rnn_4/while/maximum_iterations:output:0simple_rnn_4/time:output:0%simple_rnn_4/TensorArrayV2_1:handle:0simple_rnn_4/zeros:output:0%simple_rnn_4/strided_slice_1:output:0Dsimple_rnn_4/TensorArrayUnstack/TensorListFromTensor:output_handle:0=simple_rnn_4_simple_rnn_cell_8_matmul_readvariableop_resource>simple_rnn_4_simple_rnn_cell_8_biasadd_readvariableop_resource?simple_rnn_4_simple_rnn_cell_8_matmul_1_readvariableop_resource*
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
simple_rnn_4_while_body_100905**
cond"R 
simple_rnn_4_while_cond_100904*9
output_shapes(
&: : : : :??????????: : : : : *
parallel_iterations 2
simple_rnn_4/while?
=simple_rnn_4/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2?
=simple_rnn_4/TensorArrayV2Stack/TensorListStack/element_shape?
/simple_rnn_4/TensorArrayV2Stack/TensorListStackTensorListStacksimple_rnn_4/while:output:3Fsimple_rnn_4/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:H??????????*
element_dtype021
/simple_rnn_4/TensorArrayV2Stack/TensorListStack?
"simple_rnn_4/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2$
"simple_rnn_4/strided_slice_3/stack?
$simple_rnn_4/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2&
$simple_rnn_4/strided_slice_3/stack_1?
$simple_rnn_4/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$simple_rnn_4/strided_slice_3/stack_2?
simple_rnn_4/strided_slice_3StridedSlice8simple_rnn_4/TensorArrayV2Stack/TensorListStack:tensor:0+simple_rnn_4/strided_slice_3/stack:output:0-simple_rnn_4/strided_slice_3/stack_1:output:0-simple_rnn_4/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
simple_rnn_4/strided_slice_3?
simple_rnn_4/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
simple_rnn_4/transpose_1/perm?
simple_rnn_4/transpose_1	Transpose8simple_rnn_4/TensorArrayV2Stack/TensorListStack:tensor:0&simple_rnn_4/transpose_1/perm:output:0*
T0*,
_output_shapes
:?????????H?2
simple_rnn_4/transpose_1?
activation_10/ReluRelusimple_rnn_4/transpose_1:y:0*
T0*,
_output_shapes
:?????????H?2
activation_10/Relu?
dropout_4/IdentityIdentity activation_10/Relu:activations:0*
T0*,
_output_shapes
:?????????H?2
dropout_4/Identitys
simple_rnn_5/ShapeShapedropout_4/Identity:output:0*
T0*
_output_shapes
:2
simple_rnn_5/Shape?
 simple_rnn_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 simple_rnn_5/strided_slice/stack?
"simple_rnn_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"simple_rnn_5/strided_slice/stack_1?
"simple_rnn_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"simple_rnn_5/strided_slice/stack_2?
simple_rnn_5/strided_sliceStridedSlicesimple_rnn_5/Shape:output:0)simple_rnn_5/strided_slice/stack:output:0+simple_rnn_5/strided_slice/stack_1:output:0+simple_rnn_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
simple_rnn_5/strided_slicev
simple_rnn_5/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
simple_rnn_5/zeros/mul/y?
simple_rnn_5/zeros/mulMul#simple_rnn_5/strided_slice:output:0!simple_rnn_5/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
simple_rnn_5/zeros/muly
simple_rnn_5/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
simple_rnn_5/zeros/Less/y?
simple_rnn_5/zeros/LessLesssimple_rnn_5/zeros/mul:z:0"simple_rnn_5/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
simple_rnn_5/zeros/Less|
simple_rnn_5/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
simple_rnn_5/zeros/packed/1?
simple_rnn_5/zeros/packedPack#simple_rnn_5/strided_slice:output:0$simple_rnn_5/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
simple_rnn_5/zeros/packedy
simple_rnn_5/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
simple_rnn_5/zeros/Const?
simple_rnn_5/zerosFill"simple_rnn_5/zeros/packed:output:0!simple_rnn_5/zeros/Const:output:0*
T0*'
_output_shapes
:?????????22
simple_rnn_5/zeros?
simple_rnn_5/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
simple_rnn_5/transpose/perm?
simple_rnn_5/transpose	Transposedropout_4/Identity:output:0$simple_rnn_5/transpose/perm:output:0*
T0*,
_output_shapes
:H??????????2
simple_rnn_5/transposev
simple_rnn_5/Shape_1Shapesimple_rnn_5/transpose:y:0*
T0*
_output_shapes
:2
simple_rnn_5/Shape_1?
"simple_rnn_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"simple_rnn_5/strided_slice_1/stack?
$simple_rnn_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2&
$simple_rnn_5/strided_slice_1/stack_1?
$simple_rnn_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$simple_rnn_5/strided_slice_1/stack_2?
simple_rnn_5/strided_slice_1StridedSlicesimple_rnn_5/Shape_1:output:0+simple_rnn_5/strided_slice_1/stack:output:0-simple_rnn_5/strided_slice_1/stack_1:output:0-simple_rnn_5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
simple_rnn_5/strided_slice_1?
(simple_rnn_5/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2*
(simple_rnn_5/TensorArrayV2/element_shape?
simple_rnn_5/TensorArrayV2TensorListReserve1simple_rnn_5/TensorArrayV2/element_shape:output:0%simple_rnn_5/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
simple_rnn_5/TensorArrayV2?
Bsimple_rnn_5/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2D
Bsimple_rnn_5/TensorArrayUnstack/TensorListFromTensor/element_shape?
4simple_rnn_5/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorsimple_rnn_5/transpose:y:0Ksimple_rnn_5/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type026
4simple_rnn_5/TensorArrayUnstack/TensorListFromTensor?
"simple_rnn_5/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"simple_rnn_5/strided_slice_2/stack?
$simple_rnn_5/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2&
$simple_rnn_5/strided_slice_2/stack_1?
$simple_rnn_5/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$simple_rnn_5/strided_slice_2/stack_2?
simple_rnn_5/strided_slice_2StridedSlicesimple_rnn_5/transpose:y:0+simple_rnn_5/strided_slice_2/stack:output:0-simple_rnn_5/strided_slice_2/stack_1:output:0-simple_rnn_5/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
simple_rnn_5/strided_slice_2?
4simple_rnn_5/simple_rnn_cell_9/MatMul/ReadVariableOpReadVariableOp=simple_rnn_5_simple_rnn_cell_9_matmul_readvariableop_resource*
_output_shapes
:	?2*
dtype026
4simple_rnn_5/simple_rnn_cell_9/MatMul/ReadVariableOp?
%simple_rnn_5/simple_rnn_cell_9/MatMul	MLCMatMul%simple_rnn_5/strided_slice_2:output:0<simple_rnn_5/simple_rnn_cell_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22'
%simple_rnn_5/simple_rnn_cell_9/MatMul?
5simple_rnn_5/simple_rnn_cell_9/BiasAdd/ReadVariableOpReadVariableOp>simple_rnn_5_simple_rnn_cell_9_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype027
5simple_rnn_5/simple_rnn_cell_9/BiasAdd/ReadVariableOp?
&simple_rnn_5/simple_rnn_cell_9/BiasAddBiasAdd/simple_rnn_5/simple_rnn_cell_9/MatMul:product:0=simple_rnn_5/simple_rnn_cell_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22(
&simple_rnn_5/simple_rnn_cell_9/BiasAdd?
6simple_rnn_5/simple_rnn_cell_9/MatMul_1/ReadVariableOpReadVariableOp?simple_rnn_5_simple_rnn_cell_9_matmul_1_readvariableop_resource*
_output_shapes

:22*
dtype028
6simple_rnn_5/simple_rnn_cell_9/MatMul_1/ReadVariableOp?
'simple_rnn_5/simple_rnn_cell_9/MatMul_1	MLCMatMulsimple_rnn_5/zeros:output:0>simple_rnn_5/simple_rnn_cell_9/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22)
'simple_rnn_5/simple_rnn_cell_9/MatMul_1?
"simple_rnn_5/simple_rnn_cell_9/addAddV2/simple_rnn_5/simple_rnn_cell_9/BiasAdd:output:01simple_rnn_5/simple_rnn_cell_9/MatMul_1:product:0*
T0*'
_output_shapes
:?????????22$
"simple_rnn_5/simple_rnn_cell_9/add?
#simple_rnn_5/simple_rnn_cell_9/TanhTanh&simple_rnn_5/simple_rnn_cell_9/add:z:0*
T0*'
_output_shapes
:?????????22%
#simple_rnn_5/simple_rnn_cell_9/Tanh?
*simple_rnn_5/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   2,
*simple_rnn_5/TensorArrayV2_1/element_shape?
simple_rnn_5/TensorArrayV2_1TensorListReserve3simple_rnn_5/TensorArrayV2_1/element_shape:output:0%simple_rnn_5/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
simple_rnn_5/TensorArrayV2_1h
simple_rnn_5/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
simple_rnn_5/time?
%simple_rnn_5/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2'
%simple_rnn_5/while/maximum_iterations?
simple_rnn_5/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2!
simple_rnn_5/while/loop_counter?
simple_rnn_5/whileWhile(simple_rnn_5/while/loop_counter:output:0.simple_rnn_5/while/maximum_iterations:output:0simple_rnn_5/time:output:0%simple_rnn_5/TensorArrayV2_1:handle:0simple_rnn_5/zeros:output:0%simple_rnn_5/strided_slice_1:output:0Dsimple_rnn_5/TensorArrayUnstack/TensorListFromTensor:output_handle:0=simple_rnn_5_simple_rnn_cell_9_matmul_readvariableop_resource>simple_rnn_5_simple_rnn_cell_9_biasadd_readvariableop_resource?simple_rnn_5_simple_rnn_cell_9_matmul_1_readvariableop_resource*
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
simple_rnn_5_while_body_101015**
cond"R 
simple_rnn_5_while_cond_101014*8
output_shapes'
%: : : : :?????????2: : : : : *
parallel_iterations 2
simple_rnn_5/while?
=simple_rnn_5/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   2?
=simple_rnn_5/TensorArrayV2Stack/TensorListStack/element_shape?
/simple_rnn_5/TensorArrayV2Stack/TensorListStackTensorListStacksimple_rnn_5/while:output:3Fsimple_rnn_5/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:H?????????2*
element_dtype021
/simple_rnn_5/TensorArrayV2Stack/TensorListStack?
"simple_rnn_5/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2$
"simple_rnn_5/strided_slice_3/stack?
$simple_rnn_5/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2&
$simple_rnn_5/strided_slice_3/stack_1?
$simple_rnn_5/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$simple_rnn_5/strided_slice_3/stack_2?
simple_rnn_5/strided_slice_3StridedSlice8simple_rnn_5/TensorArrayV2Stack/TensorListStack:tensor:0+simple_rnn_5/strided_slice_3/stack:output:0-simple_rnn_5/strided_slice_3/stack_1:output:0-simple_rnn_5/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????2*
shrink_axis_mask2
simple_rnn_5/strided_slice_3?
simple_rnn_5/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
simple_rnn_5/transpose_1/perm?
simple_rnn_5/transpose_1	Transpose8simple_rnn_5/TensorArrayV2Stack/TensorListStack:tensor:0&simple_rnn_5/transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????H22
simple_rnn_5/transpose_1?
activation_11/ReluRelu%simple_rnn_5/strided_slice_3:output:0*
T0*'
_output_shapes
:?????????22
activation_11/Relu?
dropout_5/IdentityIdentity activation_11/Relu:activations:0*
T0*'
_output_shapes
:?????????22
dropout_5/Identity?
!dense_14/MLCMatMul/ReadVariableOpReadVariableOp*dense_14_mlcmatmul_readvariableop_resource*
_output_shapes

:2$*
dtype02#
!dense_14/MLCMatMul/ReadVariableOp?
dense_14/MLCMatMul	MLCMatMuldropout_5/Identity:output:0)dense_14/MLCMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????$2
dense_14/MLCMatMul?
dense_14/BiasAdd/ReadVariableOpReadVariableOp(dense_14_biasadd_readvariableop_resource*
_output_shapes
:$*
dtype02!
dense_14/BiasAdd/ReadVariableOp?
dense_14/BiasAddBiasAdddense_14/MLCMatMul:product:0'dense_14/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????$2
dense_14/BiasAdds
dense_14/TanhTanhdense_14/BiasAdd:output:0*
T0*'
_output_shapes
:?????????$2
dense_14/Tanh?
IdentityIdentitydense_14/Tanh:y:0 ^dense_14/BiasAdd/ReadVariableOp"^dense_14/MLCMatMul/ReadVariableOp6^simple_rnn_4/simple_rnn_cell_8/BiasAdd/ReadVariableOp5^simple_rnn_4/simple_rnn_cell_8/MatMul/ReadVariableOp7^simple_rnn_4/simple_rnn_cell_8/MatMul_1/ReadVariableOp^simple_rnn_4/while6^simple_rnn_5/simple_rnn_cell_9/BiasAdd/ReadVariableOp5^simple_rnn_5/simple_rnn_cell_9/MatMul/ReadVariableOp7^simple_rnn_5/simple_rnn_cell_9/MatMul_1/ReadVariableOp^simple_rnn_5/while*
T0*'
_output_shapes
:?????????$2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:?????????H::::::::2B
dense_14/BiasAdd/ReadVariableOpdense_14/BiasAdd/ReadVariableOp2F
!dense_14/MLCMatMul/ReadVariableOp!dense_14/MLCMatMul/ReadVariableOp2n
5simple_rnn_4/simple_rnn_cell_8/BiasAdd/ReadVariableOp5simple_rnn_4/simple_rnn_cell_8/BiasAdd/ReadVariableOp2l
4simple_rnn_4/simple_rnn_cell_8/MatMul/ReadVariableOp4simple_rnn_4/simple_rnn_cell_8/MatMul/ReadVariableOp2p
6simple_rnn_4/simple_rnn_cell_8/MatMul_1/ReadVariableOp6simple_rnn_4/simple_rnn_cell_8/MatMul_1/ReadVariableOp2(
simple_rnn_4/whilesimple_rnn_4/while2n
5simple_rnn_5/simple_rnn_cell_9/BiasAdd/ReadVariableOp5simple_rnn_5/simple_rnn_cell_9/BiasAdd/ReadVariableOp2l
4simple_rnn_5/simple_rnn_cell_9/MatMul/ReadVariableOp4simple_rnn_5/simple_rnn_cell_9/MatMul/ReadVariableOp2p
6simple_rnn_5/simple_rnn_cell_9/MatMul_1/ReadVariableOp6simple_rnn_5/simple_rnn_cell_9/MatMul_1/ReadVariableOp2(
simple_rnn_5/whilesimple_rnn_5/while:S O
+
_output_shapes
:?????????H
 
_user_specified_nameinputs
?
?
-__inference_simple_rnn_5_layer_call_fn_101893
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
GPU 2J 8? *P
fKRI
G__inference_simple_rnn_5_layer_call_and_return_conditional_losses_997212
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
while_cond_101177
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_101177___redundant_placeholder04
0while_while_cond_101177___redundant_placeholder14
0while_while_cond_101177___redundant_placeholder24
0while_while_cond_101177___redundant_placeholder3
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
?3
?
while_body_102062
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0<
8while_simple_rnn_cell_9_matmul_readvariableop_resource_0=
9while_simple_rnn_cell_9_biasadd_readvariableop_resource_0>
:while_simple_rnn_cell_9_matmul_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor:
6while_simple_rnn_cell_9_matmul_readvariableop_resource;
7while_simple_rnn_cell_9_biasadd_readvariableop_resource<
8while_simple_rnn_cell_9_matmul_1_readvariableop_resource??.while/simple_rnn_cell_9/BiasAdd/ReadVariableOp?-while/simple_rnn_cell_9/MatMul/ReadVariableOp?/while/simple_rnn_cell_9/MatMul_1/ReadVariableOp?
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
-while/simple_rnn_cell_9/MatMul/ReadVariableOpReadVariableOp8while_simple_rnn_cell_9_matmul_readvariableop_resource_0*
_output_shapes
:	?2*
dtype02/
-while/simple_rnn_cell_9/MatMul/ReadVariableOp?
while/simple_rnn_cell_9/MatMul	MLCMatMul0while/TensorArrayV2Read/TensorListGetItem:item:05while/simple_rnn_cell_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22 
while/simple_rnn_cell_9/MatMul?
.while/simple_rnn_cell_9/BiasAdd/ReadVariableOpReadVariableOp9while_simple_rnn_cell_9_biasadd_readvariableop_resource_0*
_output_shapes
:2*
dtype020
.while/simple_rnn_cell_9/BiasAdd/ReadVariableOp?
while/simple_rnn_cell_9/BiasAddBiasAdd(while/simple_rnn_cell_9/MatMul:product:06while/simple_rnn_cell_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22!
while/simple_rnn_cell_9/BiasAdd?
/while/simple_rnn_cell_9/MatMul_1/ReadVariableOpReadVariableOp:while_simple_rnn_cell_9_matmul_1_readvariableop_resource_0*
_output_shapes

:22*
dtype021
/while/simple_rnn_cell_9/MatMul_1/ReadVariableOp?
 while/simple_rnn_cell_9/MatMul_1	MLCMatMulwhile_placeholder_27while/simple_rnn_cell_9/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22"
 while/simple_rnn_cell_9/MatMul_1?
while/simple_rnn_cell_9/addAddV2(while/simple_rnn_cell_9/BiasAdd:output:0*while/simple_rnn_cell_9/MatMul_1:product:0*
T0*'
_output_shapes
:?????????22
while/simple_rnn_cell_9/add?
while/simple_rnn_cell_9/TanhTanhwhile/simple_rnn_cell_9/add:z:0*
T0*'
_output_shapes
:?????????22
while/simple_rnn_cell_9/Tanh?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder while/simple_rnn_cell_9/Tanh:y:0*
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
while/IdentityIdentitywhile/add_1:z:0/^while/simple_rnn_cell_9/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_9/MatMul/ReadVariableOp0^while/simple_rnn_cell_9/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations/^while/simple_rnn_cell_9/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_9/MatMul/ReadVariableOp0^while/simple_rnn_cell_9/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0/^while/simple_rnn_cell_9/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_9/MatMul/ReadVariableOp0^while/simple_rnn_cell_9/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0/^while/simple_rnn_cell_9/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_9/MatMul/ReadVariableOp0^while/simple_rnn_cell_9/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity while/simple_rnn_cell_9/Tanh:y:0/^while/simple_rnn_cell_9/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_9/MatMul/ReadVariableOp0^while/simple_rnn_cell_9/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????22
while/Identity_4")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"t
7while_simple_rnn_cell_9_biasadd_readvariableop_resource9while_simple_rnn_cell_9_biasadd_readvariableop_resource_0"v
8while_simple_rnn_cell_9_matmul_1_readvariableop_resource:while_simple_rnn_cell_9_matmul_1_readvariableop_resource_0"r
6while_simple_rnn_cell_9_matmul_readvariableop_resource8while_simple_rnn_cell_9_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :?????????2: : :::2`
.while/simple_rnn_cell_9/BiasAdd/ReadVariableOp.while/simple_rnn_cell_9/BiasAdd/ReadVariableOp2^
-while/simple_rnn_cell_9/MatMul/ReadVariableOp-while/simple_rnn_cell_9/MatMul/ReadVariableOp2b
/while/simple_rnn_cell_9/MatMul_1/ReadVariableOp/while/simple_rnn_cell_9/MatMul_1/ReadVariableOp: 
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
-__inference_simple_rnn_5_layer_call_fn_102139

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
H__inference_simple_rnn_5_layer_call_and_return_conditional_losses_1002552
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????H?:::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:?????????H?
 
_user_specified_nameinputs
?
?
while_cond_99262
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_99262___redundant_placeholder03
/while_while_cond_99262___redundant_placeholder13
/while_while_cond_99262___redundant_placeholder23
/while_while_cond_99262___redundant_placeholder3
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
?
?
.__inference_sequential_12_layer_call_fn_100589
simple_rnn_4_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallsimple_rnn_4_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????$**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_sequential_12_layer_call_and_return_conditional_losses_1005702
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????$2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:?????????H::::::::22
StatefulPartitionedCallStatefulPartitionedCall:_ [
+
_output_shapes
:?????????H
,
_user_specified_namesimple_rnn_4_input
?3
?
while_body_101178
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0<
8while_simple_rnn_cell_8_matmul_readvariableop_resource_0=
9while_simple_rnn_cell_8_biasadd_readvariableop_resource_0>
:while_simple_rnn_cell_8_matmul_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor:
6while_simple_rnn_cell_8_matmul_readvariableop_resource;
7while_simple_rnn_cell_8_biasadd_readvariableop_resource<
8while_simple_rnn_cell_8_matmul_1_readvariableop_resource??.while/simple_rnn_cell_8/BiasAdd/ReadVariableOp?-while/simple_rnn_cell_8/MatMul/ReadVariableOp?/while/simple_rnn_cell_8/MatMul_1/ReadVariableOp?
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
-while/simple_rnn_cell_8/MatMul/ReadVariableOpReadVariableOp8while_simple_rnn_cell_8_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype02/
-while/simple_rnn_cell_8/MatMul/ReadVariableOp?
while/simple_rnn_cell_8/MatMul	MLCMatMul0while/TensorArrayV2Read/TensorListGetItem:item:05while/simple_rnn_cell_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2 
while/simple_rnn_cell_8/MatMul?
.while/simple_rnn_cell_8/BiasAdd/ReadVariableOpReadVariableOp9while_simple_rnn_cell_8_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype020
.while/simple_rnn_cell_8/BiasAdd/ReadVariableOp?
while/simple_rnn_cell_8/BiasAddBiasAdd(while/simple_rnn_cell_8/MatMul:product:06while/simple_rnn_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2!
while/simple_rnn_cell_8/BiasAdd?
/while/simple_rnn_cell_8/MatMul_1/ReadVariableOpReadVariableOp:while_simple_rnn_cell_8_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype021
/while/simple_rnn_cell_8/MatMul_1/ReadVariableOp?
 while/simple_rnn_cell_8/MatMul_1	MLCMatMulwhile_placeholder_27while/simple_rnn_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2"
 while/simple_rnn_cell_8/MatMul_1?
while/simple_rnn_cell_8/addAddV2(while/simple_rnn_cell_8/BiasAdd:output:0*while/simple_rnn_cell_8/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
while/simple_rnn_cell_8/add?
while/simple_rnn_cell_8/TanhTanhwhile/simple_rnn_cell_8/add:z:0*
T0*(
_output_shapes
:??????????2
while/simple_rnn_cell_8/Tanh?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder while/simple_rnn_cell_8/Tanh:y:0*
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
while/IdentityIdentitywhile/add_1:z:0/^while/simple_rnn_cell_8/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_8/MatMul/ReadVariableOp0^while/simple_rnn_cell_8/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations/^while/simple_rnn_cell_8/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_8/MatMul/ReadVariableOp0^while/simple_rnn_cell_8/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0/^while/simple_rnn_cell_8/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_8/MatMul/ReadVariableOp0^while/simple_rnn_cell_8/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0/^while/simple_rnn_cell_8/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_8/MatMul/ReadVariableOp0^while/simple_rnn_cell_8/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity while/simple_rnn_cell_8/Tanh:y:0/^while/simple_rnn_cell_8/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_8/MatMul/ReadVariableOp0^while/simple_rnn_cell_8/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2
while/Identity_4")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"t
7while_simple_rnn_cell_8_biasadd_readvariableop_resource9while_simple_rnn_cell_8_biasadd_readvariableop_resource_0"v
8while_simple_rnn_cell_8_matmul_1_readvariableop_resource:while_simple_rnn_cell_8_matmul_1_readvariableop_resource_0"r
6while_simple_rnn_cell_8_matmul_readvariableop_resource8while_simple_rnn_cell_8_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*?
_input_shapes.
,: : : : :??????????: : :::2`
.while/simple_rnn_cell_8/BiasAdd/ReadVariableOp.while/simple_rnn_cell_8/BiasAdd/ReadVariableOp2^
-while/simple_rnn_cell_8/MatMul/ReadVariableOp-while/simple_rnn_cell_8/MatMul/ReadVariableOp2b
/while/simple_rnn_cell_8/MatMul_1/ReadVariableOp/while/simple_rnn_cell_8/MatMul_1/ReadVariableOp: 
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
?
?
M__inference_simple_rnn_cell_9_layer_call_and_return_conditional_losses_102283

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
?#
?
while_body_99263
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0#
while_simple_rnn_cell_8_99285_0#
while_simple_rnn_cell_8_99287_0#
while_simple_rnn_cell_8_99289_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor!
while_simple_rnn_cell_8_99285!
while_simple_rnn_cell_8_99287!
while_simple_rnn_cell_8_99289??/while/simple_rnn_cell_8/StatefulPartitionedCall?
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
/while/simple_rnn_cell_8/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_simple_rnn_cell_8_99285_0while_simple_rnn_cell_8_99287_0while_simple_rnn_cell_8_99289_0*
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
GPU 2J 8? *U
fPRN
L__inference_simple_rnn_cell_8_layer_call_and_return_conditional_losses_9888921
/while/simple_rnn_cell_8/StatefulPartitionedCall?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder8while/simple_rnn_cell_8/StatefulPartitionedCall:output:0*
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
while/IdentityIdentitywhile/add_1:z:00^while/simple_rnn_cell_8/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations0^while/simple_rnn_cell_8/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:00^while/simple_rnn_cell_8/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:00^while/simple_rnn_cell_8/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity8while/simple_rnn_cell_8/StatefulPartitionedCall:output:10^while/simple_rnn_cell_8/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2
while/Identity_4")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"@
while_simple_rnn_cell_8_99285while_simple_rnn_cell_8_99285_0"@
while_simple_rnn_cell_8_99287while_simple_rnn_cell_8_99287_0"@
while_simple_rnn_cell_8_99289while_simple_rnn_cell_8_99289_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*?
_input_shapes.
,: : : : :??????????: : :::2b
/while/simple_rnn_cell_8/StatefulPartitionedCall/while/simple_rnn_cell_8/StatefulPartitionedCall: 
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
?G
?
H__inference_simple_rnn_4_layer_call_and_return_conditional_losses_100074

inputs4
0simple_rnn_cell_8_matmul_readvariableop_resource5
1simple_rnn_cell_8_biasadd_readvariableop_resource6
2simple_rnn_cell_8_matmul_1_readvariableop_resource
identity??(simple_rnn_cell_8/BiasAdd/ReadVariableOp?'simple_rnn_cell_8/MatMul/ReadVariableOp?)simple_rnn_cell_8/MatMul_1/ReadVariableOp?whileD
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
:H?????????2
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
'simple_rnn_cell_8/MatMul/ReadVariableOpReadVariableOp0simple_rnn_cell_8_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02)
'simple_rnn_cell_8/MatMul/ReadVariableOp?
simple_rnn_cell_8/MatMul	MLCMatMulstrided_slice_2:output:0/simple_rnn_cell_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
simple_rnn_cell_8/MatMul?
(simple_rnn_cell_8/BiasAdd/ReadVariableOpReadVariableOp1simple_rnn_cell_8_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02*
(simple_rnn_cell_8/BiasAdd/ReadVariableOp?
simple_rnn_cell_8/BiasAddBiasAdd"simple_rnn_cell_8/MatMul:product:00simple_rnn_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
simple_rnn_cell_8/BiasAdd?
)simple_rnn_cell_8/MatMul_1/ReadVariableOpReadVariableOp2simple_rnn_cell_8_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02+
)simple_rnn_cell_8/MatMul_1/ReadVariableOp?
simple_rnn_cell_8/MatMul_1	MLCMatMulzeros:output:01simple_rnn_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
simple_rnn_cell_8/MatMul_1?
simple_rnn_cell_8/addAddV2"simple_rnn_cell_8/BiasAdd:output:0$simple_rnn_cell_8/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
simple_rnn_cell_8/add?
simple_rnn_cell_8/TanhTanhsimple_rnn_cell_8/add:z:0*
T0*(
_output_shapes
:??????????2
simple_rnn_cell_8/Tanh?
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:00simple_rnn_cell_8_matmul_readvariableop_resource1simple_rnn_cell_8_biasadd_readvariableop_resource2simple_rnn_cell_8_matmul_1_readvariableop_resource*
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
while_body_100008*
condR
while_cond_100007*9
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
:H??????????*
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
:?????????H?2
transpose_1?
IdentityIdentitytranspose_1:y:0)^simple_rnn_cell_8/BiasAdd/ReadVariableOp(^simple_rnn_cell_8/MatMul/ReadVariableOp*^simple_rnn_cell_8/MatMul_1/ReadVariableOp^while*
T0*,
_output_shapes
:?????????H?2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????H:::2T
(simple_rnn_cell_8/BiasAdd/ReadVariableOp(simple_rnn_cell_8/BiasAdd/ReadVariableOp2R
'simple_rnn_cell_8/MatMul/ReadVariableOp'simple_rnn_cell_8/MatMul/ReadVariableOp2V
)simple_rnn_cell_8/MatMul_1/ReadVariableOp)simple_rnn_cell_8/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:?????????H
 
_user_specified_nameinputs
?3
?
while_body_101290
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0<
8while_simple_rnn_cell_8_matmul_readvariableop_resource_0=
9while_simple_rnn_cell_8_biasadd_readvariableop_resource_0>
:while_simple_rnn_cell_8_matmul_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor:
6while_simple_rnn_cell_8_matmul_readvariableop_resource;
7while_simple_rnn_cell_8_biasadd_readvariableop_resource<
8while_simple_rnn_cell_8_matmul_1_readvariableop_resource??.while/simple_rnn_cell_8/BiasAdd/ReadVariableOp?-while/simple_rnn_cell_8/MatMul/ReadVariableOp?/while/simple_rnn_cell_8/MatMul_1/ReadVariableOp?
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
-while/simple_rnn_cell_8/MatMul/ReadVariableOpReadVariableOp8while_simple_rnn_cell_8_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype02/
-while/simple_rnn_cell_8/MatMul/ReadVariableOp?
while/simple_rnn_cell_8/MatMul	MLCMatMul0while/TensorArrayV2Read/TensorListGetItem:item:05while/simple_rnn_cell_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2 
while/simple_rnn_cell_8/MatMul?
.while/simple_rnn_cell_8/BiasAdd/ReadVariableOpReadVariableOp9while_simple_rnn_cell_8_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype020
.while/simple_rnn_cell_8/BiasAdd/ReadVariableOp?
while/simple_rnn_cell_8/BiasAddBiasAdd(while/simple_rnn_cell_8/MatMul:product:06while/simple_rnn_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2!
while/simple_rnn_cell_8/BiasAdd?
/while/simple_rnn_cell_8/MatMul_1/ReadVariableOpReadVariableOp:while_simple_rnn_cell_8_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype021
/while/simple_rnn_cell_8/MatMul_1/ReadVariableOp?
 while/simple_rnn_cell_8/MatMul_1	MLCMatMulwhile_placeholder_27while/simple_rnn_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2"
 while/simple_rnn_cell_8/MatMul_1?
while/simple_rnn_cell_8/addAddV2(while/simple_rnn_cell_8/BiasAdd:output:0*while/simple_rnn_cell_8/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
while/simple_rnn_cell_8/add?
while/simple_rnn_cell_8/TanhTanhwhile/simple_rnn_cell_8/add:z:0*
T0*(
_output_shapes
:??????????2
while/simple_rnn_cell_8/Tanh?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder while/simple_rnn_cell_8/Tanh:y:0*
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
while/IdentityIdentitywhile/add_1:z:0/^while/simple_rnn_cell_8/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_8/MatMul/ReadVariableOp0^while/simple_rnn_cell_8/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations/^while/simple_rnn_cell_8/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_8/MatMul/ReadVariableOp0^while/simple_rnn_cell_8/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0/^while/simple_rnn_cell_8/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_8/MatMul/ReadVariableOp0^while/simple_rnn_cell_8/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0/^while/simple_rnn_cell_8/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_8/MatMul/ReadVariableOp0^while/simple_rnn_cell_8/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity while/simple_rnn_cell_8/Tanh:y:0/^while/simple_rnn_cell_8/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_8/MatMul/ReadVariableOp0^while/simple_rnn_cell_8/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2
while/Identity_4")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"t
7while_simple_rnn_cell_8_biasadd_readvariableop_resource9while_simple_rnn_cell_8_biasadd_readvariableop_resource_0"v
8while_simple_rnn_cell_8_matmul_1_readvariableop_resource:while_simple_rnn_cell_8_matmul_1_readvariableop_resource_0"r
6while_simple_rnn_cell_8_matmul_readvariableop_resource8while_simple_rnn_cell_8_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*?
_input_shapes.
,: : : : :??????????: : :::2`
.while/simple_rnn_cell_8/BiasAdd/ReadVariableOp.while/simple_rnn_cell_8/BiasAdd/ReadVariableOp2^
-while/simple_rnn_cell_8/MatMul/ReadVariableOp-while/simple_rnn_cell_8/MatMul/ReadVariableOp2b
/while/simple_rnn_cell_8/MatMul_1/ReadVariableOp/while/simple_rnn_cell_8/MatMul_1/ReadVariableOp: 
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
?3
?
while_body_101950
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0<
8while_simple_rnn_cell_9_matmul_readvariableop_resource_0=
9while_simple_rnn_cell_9_biasadd_readvariableop_resource_0>
:while_simple_rnn_cell_9_matmul_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor:
6while_simple_rnn_cell_9_matmul_readvariableop_resource;
7while_simple_rnn_cell_9_biasadd_readvariableop_resource<
8while_simple_rnn_cell_9_matmul_1_readvariableop_resource??.while/simple_rnn_cell_9/BiasAdd/ReadVariableOp?-while/simple_rnn_cell_9/MatMul/ReadVariableOp?/while/simple_rnn_cell_9/MatMul_1/ReadVariableOp?
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
-while/simple_rnn_cell_9/MatMul/ReadVariableOpReadVariableOp8while_simple_rnn_cell_9_matmul_readvariableop_resource_0*
_output_shapes
:	?2*
dtype02/
-while/simple_rnn_cell_9/MatMul/ReadVariableOp?
while/simple_rnn_cell_9/MatMul	MLCMatMul0while/TensorArrayV2Read/TensorListGetItem:item:05while/simple_rnn_cell_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22 
while/simple_rnn_cell_9/MatMul?
.while/simple_rnn_cell_9/BiasAdd/ReadVariableOpReadVariableOp9while_simple_rnn_cell_9_biasadd_readvariableop_resource_0*
_output_shapes
:2*
dtype020
.while/simple_rnn_cell_9/BiasAdd/ReadVariableOp?
while/simple_rnn_cell_9/BiasAddBiasAdd(while/simple_rnn_cell_9/MatMul:product:06while/simple_rnn_cell_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22!
while/simple_rnn_cell_9/BiasAdd?
/while/simple_rnn_cell_9/MatMul_1/ReadVariableOpReadVariableOp:while_simple_rnn_cell_9_matmul_1_readvariableop_resource_0*
_output_shapes

:22*
dtype021
/while/simple_rnn_cell_9/MatMul_1/ReadVariableOp?
 while/simple_rnn_cell_9/MatMul_1	MLCMatMulwhile_placeholder_27while/simple_rnn_cell_9/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22"
 while/simple_rnn_cell_9/MatMul_1?
while/simple_rnn_cell_9/addAddV2(while/simple_rnn_cell_9/BiasAdd:output:0*while/simple_rnn_cell_9/MatMul_1:product:0*
T0*'
_output_shapes
:?????????22
while/simple_rnn_cell_9/add?
while/simple_rnn_cell_9/TanhTanhwhile/simple_rnn_cell_9/add:z:0*
T0*'
_output_shapes
:?????????22
while/simple_rnn_cell_9/Tanh?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder while/simple_rnn_cell_9/Tanh:y:0*
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
while/IdentityIdentitywhile/add_1:z:0/^while/simple_rnn_cell_9/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_9/MatMul/ReadVariableOp0^while/simple_rnn_cell_9/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations/^while/simple_rnn_cell_9/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_9/MatMul/ReadVariableOp0^while/simple_rnn_cell_9/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0/^while/simple_rnn_cell_9/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_9/MatMul/ReadVariableOp0^while/simple_rnn_cell_9/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0/^while/simple_rnn_cell_9/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_9/MatMul/ReadVariableOp0^while/simple_rnn_cell_9/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity while/simple_rnn_cell_9/Tanh:y:0/^while/simple_rnn_cell_9/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_9/MatMul/ReadVariableOp0^while/simple_rnn_cell_9/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????22
while/Identity_4")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"t
7while_simple_rnn_cell_9_biasadd_readvariableop_resource9while_simple_rnn_cell_9_biasadd_readvariableop_resource_0"v
8while_simple_rnn_cell_9_matmul_1_readvariableop_resource:while_simple_rnn_cell_9_matmul_1_readvariableop_resource_0"r
6while_simple_rnn_cell_9_matmul_readvariableop_resource8while_simple_rnn_cell_9_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :?????????2: : :::2`
.while/simple_rnn_cell_9/BiasAdd/ReadVariableOp.while/simple_rnn_cell_9/BiasAdd/ReadVariableOp2^
-while/simple_rnn_cell_9/MatMul/ReadVariableOp-while/simple_rnn_cell_9/MatMul/ReadVariableOp2b
/while/simple_rnn_cell_9/MatMul_1/ReadVariableOp/while/simple_rnn_cell_9/MatMul_1/ReadVariableOp: 
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
?
while_cond_101815
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_101815___redundant_placeholder04
0while_while_cond_101815___redundant_placeholder14
0while_while_cond_101815___redundant_placeholder24
0while_while_cond_101815___redundant_placeholder3
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
:"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
U
simple_rnn_4_input?
$serving_default_simple_rnn_4_input:0?????????H<
dense_140
StatefulPartitionedCall:0?????????$tensorflow/serving/predict:Ӹ
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
_tf_keras_sequential?3{"class_name": "Sequential", "name": "sequential_12", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_12", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 72, 5]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "simple_rnn_4_input"}}, {"class_name": "SimpleRNN", "config": {"name": "simple_rnn_4", "trainable": true, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 150, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0}}, {"class_name": "Activation", "config": {"name": "activation_10", "trainable": true, "dtype": "float32", "activation": "relu"}}, {"class_name": "Dropout", "config": {"name": "dropout_4", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}, {"class_name": "SimpleRNN", "config": {"name": "simple_rnn_5", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 50, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0}}, {"class_name": "Activation", "config": {"name": "activation_11", "trainable": true, "dtype": "float32", "activation": "relu"}}, {"class_name": "Dropout", "config": {"name": "dropout_5", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_14", "trainable": true, "dtype": "float32", "units": 36, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 5]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 72, 5]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_12", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 72, 5]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "simple_rnn_4_input"}}, {"class_name": "SimpleRNN", "config": {"name": "simple_rnn_4", "trainable": true, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 150, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0}}, {"class_name": "Activation", "config": {"name": "activation_10", "trainable": true, "dtype": "float32", "activation": "relu"}}, {"class_name": "Dropout", "config": {"name": "dropout_4", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}, {"class_name": "SimpleRNN", "config": {"name": "simple_rnn_5", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 50, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0}}, {"class_name": "Activation", "config": {"name": "activation_11", "trainable": true, "dtype": "float32", "activation": "relu"}}, {"class_name": "Dropout", "config": {"name": "dropout_5", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_14", "trainable": true, "dtype": "float32", "units": 36, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": "mean_squared_error", "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "mae", "dtype": "float32", "fn": "mean_absolute_error"}}, {"class_name": "MeanMetricWrapper", "config": {"name": "acc", "dtype": "float32", "fn": "categorical_accuracy"}}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
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
_tf_keras_rnn_layer?	{"class_name": "SimpleRNN", "name": "simple_rnn_4", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "simple_rnn_4", "trainable": true, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 150, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 5]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 72, 5]}}
?
regularization_losses
	variables
trainable_variables
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Activation", "name": "activation_10", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_10", "trainable": true, "dtype": "float32", "activation": "relu"}}
?
regularization_losses
	variables
trainable_variables
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout_4", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_4", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}
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
_tf_keras_rnn_layer?	{"class_name": "SimpleRNN", "name": "simple_rnn_5", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "simple_rnn_5", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 50, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 150]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 72, 150]}}
?
"regularization_losses
#	variables
$trainable_variables
%	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Activation", "name": "activation_11", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_11", "trainable": true, "dtype": "float32", "activation": "relu"}}
?
&regularization_losses
'	variables
(trainable_variables
)	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout_5", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_5", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}
?

*kernel
+bias
,regularization_losses
-	variables
.trainable_variables
/	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_14", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_14", "trainable": true, "dtype": "float32", "units": 36, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 50}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50]}}
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
_tf_keras_layer?{"class_name": "SimpleRNNCell", "name": "simple_rnn_cell_8", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "simple_rnn_cell_8", "trainable": true, "dtype": "float32", "units": 150, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0}}
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
_tf_keras_layer?{"class_name": "SimpleRNNCell", "name": "simple_rnn_cell_9", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "simple_rnn_cell_9", "trainable": true, "dtype": "float32", "units": 50, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0}}
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
!:2$2dense_14/kernel
:$2dense_14/bias
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
8:6	?2%simple_rnn_4/simple_rnn_cell_8/kernel
C:A
??2/simple_rnn_4/simple_rnn_cell_8/recurrent_kernel
2:0?2#simple_rnn_4/simple_rnn_cell_8/bias
8:6	?22%simple_rnn_5/simple_rnn_cell_9/kernel
A:?222/simple_rnn_5/simple_rnn_cell_9/recurrent_kernel
1:/22#simple_rnn_5/simple_rnn_cell_9/bias
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
&:$2$2Adam/dense_14/kernel/m
 :$2Adam/dense_14/bias/m
=:;	?2,Adam/simple_rnn_4/simple_rnn_cell_8/kernel/m
H:F
??26Adam/simple_rnn_4/simple_rnn_cell_8/recurrent_kernel/m
7:5?2*Adam/simple_rnn_4/simple_rnn_cell_8/bias/m
=:;	?22,Adam/simple_rnn_5/simple_rnn_cell_9/kernel/m
F:D2226Adam/simple_rnn_5/simple_rnn_cell_9/recurrent_kernel/m
6:422*Adam/simple_rnn_5/simple_rnn_cell_9/bias/m
&:$2$2Adam/dense_14/kernel/v
 :$2Adam/dense_14/bias/v
=:;	?2,Adam/simple_rnn_4/simple_rnn_cell_8/kernel/v
H:F
??26Adam/simple_rnn_4/simple_rnn_cell_8/recurrent_kernel/v
7:5?2*Adam/simple_rnn_4/simple_rnn_cell_8/bias/v
=:;	?22,Adam/simple_rnn_5/simple_rnn_cell_9/kernel/v
F:D2226Adam/simple_rnn_5/simple_rnn_cell_9/recurrent_kernel/v
6:422*Adam/simple_rnn_5/simple_rnn_cell_9/bias/v
?2?
.__inference_sequential_12_layer_call_fn_101132
.__inference_sequential_12_layer_call_fn_100541
.__inference_sequential_12_layer_call_fn_101111
.__inference_sequential_12_layer_call_fn_100589?
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
 __inference__wrapped_model_98823?
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
simple_rnn_4_input?????????H
?2?
I__inference_sequential_12_layer_call_and_return_conditional_losses_101090
I__inference_sequential_12_layer_call_and_return_conditional_losses_100492
I__inference_sequential_12_layer_call_and_return_conditional_losses_100859
I__inference_sequential_12_layer_call_and_return_conditional_losses_100465?
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
-__inference_simple_rnn_4_layer_call_fn_101624
-__inference_simple_rnn_4_layer_call_fn_101367
-__inference_simple_rnn_4_layer_call_fn_101613
-__inference_simple_rnn_4_layer_call_fn_101378?
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
H__inference_simple_rnn_4_layer_call_and_return_conditional_losses_101356
H__inference_simple_rnn_4_layer_call_and_return_conditional_losses_101490
H__inference_simple_rnn_4_layer_call_and_return_conditional_losses_101244
H__inference_simple_rnn_4_layer_call_and_return_conditional_losses_101602?
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
.__inference_activation_10_layer_call_fn_101634?
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
I__inference_activation_10_layer_call_and_return_conditional_losses_101629?
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
*__inference_dropout_4_layer_call_fn_101653
*__inference_dropout_4_layer_call_fn_101658?
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
E__inference_dropout_4_layer_call_and_return_conditional_losses_101648
E__inference_dropout_4_layer_call_and_return_conditional_losses_101643?
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
-__inference_simple_rnn_5_layer_call_fn_101893
-__inference_simple_rnn_5_layer_call_fn_102139
-__inference_simple_rnn_5_layer_call_fn_102150
-__inference_simple_rnn_5_layer_call_fn_101904?
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
H__inference_simple_rnn_5_layer_call_and_return_conditional_losses_102016
H__inference_simple_rnn_5_layer_call_and_return_conditional_losses_101882
H__inference_simple_rnn_5_layer_call_and_return_conditional_losses_101770
H__inference_simple_rnn_5_layer_call_and_return_conditional_losses_102128?
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
.__inference_activation_11_layer_call_fn_102160?
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
I__inference_activation_11_layer_call_and_return_conditional_losses_102155?
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
*__inference_dropout_5_layer_call_fn_102179
*__inference_dropout_5_layer_call_fn_102184?
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
E__inference_dropout_5_layer_call_and_return_conditional_losses_102169
E__inference_dropout_5_layer_call_and_return_conditional_losses_102174?
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
)__inference_dense_14_layer_call_fn_102204?
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
D__inference_dense_14_layer_call_and_return_conditional_losses_102195?
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
$__inference_signature_wrapper_100620simple_rnn_4_input"?
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
2__inference_simple_rnn_cell_8_layer_call_fn_102266
2__inference_simple_rnn_cell_8_layer_call_fn_102252?
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
M__inference_simple_rnn_cell_8_layer_call_and_return_conditional_losses_102221
M__inference_simple_rnn_cell_8_layer_call_and_return_conditional_losses_102238?
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
2__inference_simple_rnn_cell_9_layer_call_fn_102314
2__inference_simple_rnn_cell_9_layer_call_fn_102328?
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
M__inference_simple_rnn_cell_9_layer_call_and_return_conditional_losses_102283
M__inference_simple_rnn_cell_9_layer_call_and_return_conditional_losses_102300?
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
 __inference__wrapped_model_98823?5768:9*+??<
5?2
0?-
simple_rnn_4_input?????????H
? "3?0
.
dense_14"?
dense_14?????????$?
I__inference_activation_10_layer_call_and_return_conditional_losses_101629b4?1
*?'
%?"
inputs?????????H?
? "*?'
 ?
0?????????H?
? ?
.__inference_activation_10_layer_call_fn_101634U4?1
*?'
%?"
inputs?????????H?
? "??????????H??
I__inference_activation_11_layer_call_and_return_conditional_losses_102155X/?,
%?"
 ?
inputs?????????2
? "%?"
?
0?????????2
? }
.__inference_activation_11_layer_call_fn_102160K/?,
%?"
 ?
inputs?????????2
? "??????????2?
D__inference_dense_14_layer_call_and_return_conditional_losses_102195\*+/?,
%?"
 ?
inputs?????????2
? "%?"
?
0?????????$
? |
)__inference_dense_14_layer_call_fn_102204O*+/?,
%?"
 ?
inputs?????????2
? "??????????$?
E__inference_dropout_4_layer_call_and_return_conditional_losses_101643f8?5
.?+
%?"
inputs?????????H?
p
? "*?'
 ?
0?????????H?
? ?
E__inference_dropout_4_layer_call_and_return_conditional_losses_101648f8?5
.?+
%?"
inputs?????????H?
p 
? "*?'
 ?
0?????????H?
? ?
*__inference_dropout_4_layer_call_fn_101653Y8?5
.?+
%?"
inputs?????????H?
p
? "??????????H??
*__inference_dropout_4_layer_call_fn_101658Y8?5
.?+
%?"
inputs?????????H?
p 
? "??????????H??
E__inference_dropout_5_layer_call_and_return_conditional_losses_102169\3?0
)?&
 ?
inputs?????????2
p
? "%?"
?
0?????????2
? ?
E__inference_dropout_5_layer_call_and_return_conditional_losses_102174\3?0
)?&
 ?
inputs?????????2
p 
? "%?"
?
0?????????2
? }
*__inference_dropout_5_layer_call_fn_102179O3?0
)?&
 ?
inputs?????????2
p
? "??????????2}
*__inference_dropout_5_layer_call_fn_102184O3?0
)?&
 ?
inputs?????????2
p 
? "??????????2?
I__inference_sequential_12_layer_call_and_return_conditional_losses_100465z5768:9*+G?D
=?:
0?-
simple_rnn_4_input?????????H
p

 
? "%?"
?
0?????????$
? ?
I__inference_sequential_12_layer_call_and_return_conditional_losses_100492z5768:9*+G?D
=?:
0?-
simple_rnn_4_input?????????H
p 

 
? "%?"
?
0?????????$
? ?
I__inference_sequential_12_layer_call_and_return_conditional_losses_100859n5768:9*+;?8
1?.
$?!
inputs?????????H
p

 
? "%?"
?
0?????????$
? ?
I__inference_sequential_12_layer_call_and_return_conditional_losses_101090n5768:9*+;?8
1?.
$?!
inputs?????????H
p 

 
? "%?"
?
0?????????$
? ?
.__inference_sequential_12_layer_call_fn_100541m5768:9*+G?D
=?:
0?-
simple_rnn_4_input?????????H
p

 
? "??????????$?
.__inference_sequential_12_layer_call_fn_100589m5768:9*+G?D
=?:
0?-
simple_rnn_4_input?????????H
p 

 
? "??????????$?
.__inference_sequential_12_layer_call_fn_101111a5768:9*+;?8
1?.
$?!
inputs?????????H
p

 
? "??????????$?
.__inference_sequential_12_layer_call_fn_101132a5768:9*+;?8
1?.
$?!
inputs?????????H
p 

 
? "??????????$?
$__inference_signature_wrapper_100620?5768:9*+U?R
? 
K?H
F
simple_rnn_4_input0?-
simple_rnn_4_input?????????H"3?0
.
dense_14"?
dense_14?????????$?
H__inference_simple_rnn_4_layer_call_and_return_conditional_losses_101244r576??<
5?2
$?!
inputs?????????H

 
p

 
? "*?'
 ?
0?????????H?
? ?
H__inference_simple_rnn_4_layer_call_and_return_conditional_losses_101356r576??<
5?2
$?!
inputs?????????H

 
p 

 
? "*?'
 ?
0?????????H?
? ?
H__inference_simple_rnn_4_layer_call_and_return_conditional_losses_101490?576O?L
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
H__inference_simple_rnn_4_layer_call_and_return_conditional_losses_101602?576O?L
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
-__inference_simple_rnn_4_layer_call_fn_101367e576??<
5?2
$?!
inputs?????????H

 
p

 
? "??????????H??
-__inference_simple_rnn_4_layer_call_fn_101378e576??<
5?2
$?!
inputs?????????H

 
p 

 
? "??????????H??
-__inference_simple_rnn_4_layer_call_fn_101613~576O?L
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
-__inference_simple_rnn_4_layer_call_fn_101624~576O?L
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
H__inference_simple_rnn_5_layer_call_and_return_conditional_losses_101770~8:9P?M
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
H__inference_simple_rnn_5_layer_call_and_return_conditional_losses_101882~8:9P?M
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
H__inference_simple_rnn_5_layer_call_and_return_conditional_losses_102016n8:9@?=
6?3
%?"
inputs?????????H?

 
p

 
? "%?"
?
0?????????2
? ?
H__inference_simple_rnn_5_layer_call_and_return_conditional_losses_102128n8:9@?=
6?3
%?"
inputs?????????H?

 
p 

 
? "%?"
?
0?????????2
? ?
-__inference_simple_rnn_5_layer_call_fn_101893q8:9P?M
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
-__inference_simple_rnn_5_layer_call_fn_101904q8:9P?M
F?C
5?2
0?-
inputs/0???????????????????

 
p 

 
? "??????????2?
-__inference_simple_rnn_5_layer_call_fn_102139a8:9@?=
6?3
%?"
inputs?????????H?

 
p

 
? "??????????2?
-__inference_simple_rnn_5_layer_call_fn_102150a8:9@?=
6?3
%?"
inputs?????????H?

 
p 

 
? "??????????2?
M__inference_simple_rnn_cell_8_layer_call_and_return_conditional_losses_102221?576]?Z
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
M__inference_simple_rnn_cell_8_layer_call_and_return_conditional_losses_102238?576]?Z
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
2__inference_simple_rnn_cell_8_layer_call_fn_102252?576]?Z
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
2__inference_simple_rnn_cell_8_layer_call_fn_102266?576]?Z
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
M__inference_simple_rnn_cell_9_layer_call_and_return_conditional_losses_102283?8:9]?Z
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
M__inference_simple_rnn_cell_9_layer_call_and_return_conditional_losses_102300?8:9]?Z
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
2__inference_simple_rnn_cell_9_layer_call_fn_102314?8:9]?Z
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
2__inference_simple_rnn_cell_9_layer_call_fn_102328?8:9]?Z
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