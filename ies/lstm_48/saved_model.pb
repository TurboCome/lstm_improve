ݟ
??
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
MLCLSTM
x"T
hidden_size
output_size
kernel"T
recurrent_kernel"T	
bias"T
output"T"
Ttype:
2"
return_sequencesbool( "
dropoutfloat"
bidirectionalbool( "

activationstringtanh
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
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
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
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*	2.4.0-rc02&tf_macos-v0.1-alpha2-AS-67-gf3595294ab8??
{
dense_22/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?* 
shared_namedense_22/kernel
t
#dense_22/kernel/Read/ReadVariableOpReadVariableOpdense_22/kernel*
_output_shapes
:	?*
dtype0
r
dense_22/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_22/bias
k
!dense_22/bias/Read/ReadVariableOpReadVariableOpdense_22/bias*
_output_shapes
:*
dtype0
z
dense_23/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:20* 
shared_namedense_23/kernel
s
#dense_23/kernel/Read/ReadVariableOpReadVariableOpdense_23/kernel*
_output_shapes

:20*
dtype0
r
dense_23/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*
shared_namedense_23/bias
k
!dense_23/bias/Read/ReadVariableOpReadVariableOpdense_23/bias*
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
lstm_12/lstm_cell_24/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*,
shared_namelstm_12/lstm_cell_24/kernel
?
/lstm_12/lstm_cell_24/kernel/Read/ReadVariableOpReadVariableOplstm_12/lstm_cell_24/kernel*
_output_shapes
:	?*
dtype0
?
%lstm_12/lstm_cell_24/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*6
shared_name'%lstm_12/lstm_cell_24/recurrent_kernel
?
9lstm_12/lstm_cell_24/recurrent_kernel/Read/ReadVariableOpReadVariableOp%lstm_12/lstm_cell_24/recurrent_kernel* 
_output_shapes
:
??*
dtype0
?
lstm_12/lstm_cell_24/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?**
shared_namelstm_12/lstm_cell_24/bias
?
-lstm_12/lstm_cell_24/bias/Read/ReadVariableOpReadVariableOplstm_12/lstm_cell_24/bias*
_output_shapes	
:?*
dtype0
?
lstm_13/lstm_cell_25/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*,
shared_namelstm_13/lstm_cell_25/kernel
?
/lstm_13/lstm_cell_25/kernel/Read/ReadVariableOpReadVariableOplstm_13/lstm_cell_25/kernel*
_output_shapes
:	?*
dtype0
?
%lstm_13/lstm_cell_25/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2?*6
shared_name'%lstm_13/lstm_cell_25/recurrent_kernel
?
9lstm_13/lstm_cell_25/recurrent_kernel/Read/ReadVariableOpReadVariableOp%lstm_13/lstm_cell_25/recurrent_kernel*
_output_shapes
:	2?*
dtype0
?
lstm_13/lstm_cell_25/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?**
shared_namelstm_13/lstm_cell_25/bias
?
-lstm_13/lstm_cell_25/bias/Read/ReadVariableOpReadVariableOplstm_13/lstm_cell_25/bias*
_output_shapes	
:?*
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
Adam/dense_22/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*'
shared_nameAdam/dense_22/kernel/m
?
*Adam/dense_22/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_22/kernel/m*
_output_shapes
:	?*
dtype0
?
Adam/dense_22/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_22/bias/m
y
(Adam/dense_22/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_22/bias/m*
_output_shapes
:*
dtype0
?
Adam/dense_23/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:20*'
shared_nameAdam/dense_23/kernel/m
?
*Adam/dense_23/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_23/kernel/m*
_output_shapes

:20*
dtype0
?
Adam/dense_23/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*%
shared_nameAdam/dense_23/bias/m
y
(Adam/dense_23/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_23/bias/m*
_output_shapes
:0*
dtype0
?
"Adam/lstm_12/lstm_cell_24/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*3
shared_name$"Adam/lstm_12/lstm_cell_24/kernel/m
?
6Adam/lstm_12/lstm_cell_24/kernel/m/Read/ReadVariableOpReadVariableOp"Adam/lstm_12/lstm_cell_24/kernel/m*
_output_shapes
:	?*
dtype0
?
,Adam/lstm_12/lstm_cell_24/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*=
shared_name.,Adam/lstm_12/lstm_cell_24/recurrent_kernel/m
?
@Adam/lstm_12/lstm_cell_24/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp,Adam/lstm_12/lstm_cell_24/recurrent_kernel/m* 
_output_shapes
:
??*
dtype0
?
 Adam/lstm_12/lstm_cell_24/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*1
shared_name" Adam/lstm_12/lstm_cell_24/bias/m
?
4Adam/lstm_12/lstm_cell_24/bias/m/Read/ReadVariableOpReadVariableOp Adam/lstm_12/lstm_cell_24/bias/m*
_output_shapes	
:?*
dtype0
?
"Adam/lstm_13/lstm_cell_25/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*3
shared_name$"Adam/lstm_13/lstm_cell_25/kernel/m
?
6Adam/lstm_13/lstm_cell_25/kernel/m/Read/ReadVariableOpReadVariableOp"Adam/lstm_13/lstm_cell_25/kernel/m*
_output_shapes
:	?*
dtype0
?
,Adam/lstm_13/lstm_cell_25/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2?*=
shared_name.,Adam/lstm_13/lstm_cell_25/recurrent_kernel/m
?
@Adam/lstm_13/lstm_cell_25/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp,Adam/lstm_13/lstm_cell_25/recurrent_kernel/m*
_output_shapes
:	2?*
dtype0
?
 Adam/lstm_13/lstm_cell_25/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*1
shared_name" Adam/lstm_13/lstm_cell_25/bias/m
?
4Adam/lstm_13/lstm_cell_25/bias/m/Read/ReadVariableOpReadVariableOp Adam/lstm_13/lstm_cell_25/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/dense_22/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*'
shared_nameAdam/dense_22/kernel/v
?
*Adam/dense_22/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_22/kernel/v*
_output_shapes
:	?*
dtype0
?
Adam/dense_22/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_22/bias/v
y
(Adam/dense_22/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_22/bias/v*
_output_shapes
:*
dtype0
?
Adam/dense_23/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:20*'
shared_nameAdam/dense_23/kernel/v
?
*Adam/dense_23/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_23/kernel/v*
_output_shapes

:20*
dtype0
?
Adam/dense_23/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*%
shared_nameAdam/dense_23/bias/v
y
(Adam/dense_23/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_23/bias/v*
_output_shapes
:0*
dtype0
?
"Adam/lstm_12/lstm_cell_24/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*3
shared_name$"Adam/lstm_12/lstm_cell_24/kernel/v
?
6Adam/lstm_12/lstm_cell_24/kernel/v/Read/ReadVariableOpReadVariableOp"Adam/lstm_12/lstm_cell_24/kernel/v*
_output_shapes
:	?*
dtype0
?
,Adam/lstm_12/lstm_cell_24/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*=
shared_name.,Adam/lstm_12/lstm_cell_24/recurrent_kernel/v
?
@Adam/lstm_12/lstm_cell_24/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp,Adam/lstm_12/lstm_cell_24/recurrent_kernel/v* 
_output_shapes
:
??*
dtype0
?
 Adam/lstm_12/lstm_cell_24/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*1
shared_name" Adam/lstm_12/lstm_cell_24/bias/v
?
4Adam/lstm_12/lstm_cell_24/bias/v/Read/ReadVariableOpReadVariableOp Adam/lstm_12/lstm_cell_24/bias/v*
_output_shapes	
:?*
dtype0
?
"Adam/lstm_13/lstm_cell_25/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*3
shared_name$"Adam/lstm_13/lstm_cell_25/kernel/v
?
6Adam/lstm_13/lstm_cell_25/kernel/v/Read/ReadVariableOpReadVariableOp"Adam/lstm_13/lstm_cell_25/kernel/v*
_output_shapes
:	?*
dtype0
?
,Adam/lstm_13/lstm_cell_25/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2?*=
shared_name.,Adam/lstm_13/lstm_cell_25/recurrent_kernel/v
?
@Adam/lstm_13/lstm_cell_25/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp,Adam/lstm_13/lstm_cell_25/recurrent_kernel/v*
_output_shapes
:	2?*
dtype0
?
 Adam/lstm_13/lstm_cell_25/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*1
shared_name" Adam/lstm_13/lstm_cell_25/bias/v
?
4Adam/lstm_13/lstm_cell_25/bias/v/Read/ReadVariableOpReadVariableOp Adam/lstm_13/lstm_cell_25/bias/v*
_output_shapes	
:?*
dtype0

NoOpNoOp
?<
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?;
value?;B?; B?;
?
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
	optimizer
regularization_losses
	variables
trainable_variables
		keras_api


signatures
l
cell

state_spec
regularization_losses
	variables
trainable_variables
	keras_api
h

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
l
cell

state_spec
regularization_losses
	variables
trainable_variables
	keras_api
h

kernel
bias
regularization_losses
 	variables
!trainable_variables
"	keras_api
?
#iter

$beta_1

%beta_2
	&decay
'learning_ratemlmmmnmo(mp)mq*mr+ms,mt-muvvvwvxvy(vz)v{*v|+v},v~-v
 
F
(0
)1
*2
3
4
+5
,6
-7
8
9
F
(0
)1
*2
3
4
+5
,6
-7
8
9
?
regularization_losses
.layer_regularization_losses
/layer_metrics
	variables
0non_trainable_variables
trainable_variables
1metrics

2layers
 
~

(kernel
)recurrent_kernel
*bias
3regularization_losses
4	variables
5trainable_variables
6	keras_api
 
 

(0
)1
*2

(0
)1
*2
?
regularization_losses
7layer_regularization_losses
8layer_metrics
	variables
9non_trainable_variables
trainable_variables
:metrics

;layers

<states
[Y
VARIABLE_VALUEdense_22/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_22/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
?
regularization_losses
=layer_regularization_losses
>layer_metrics
	variables
?non_trainable_variables
trainable_variables
@metrics

Alayers
~

+kernel
,recurrent_kernel
-bias
Bregularization_losses
C	variables
Dtrainable_variables
E	keras_api
 
 

+0
,1
-2

+0
,1
-2
?
regularization_losses
Flayer_regularization_losses
Glayer_metrics
	variables
Hnon_trainable_variables
trainable_variables
Imetrics

Jlayers

Kstates
[Y
VARIABLE_VALUEdense_23/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_23/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
?
regularization_losses
Llayer_regularization_losses
Mlayer_metrics
 	variables
Nnon_trainable_variables
!trainable_variables
Ometrics

Players
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
WU
VARIABLE_VALUElstm_12/lstm_cell_24/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUE%lstm_12/lstm_cell_24/recurrent_kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUElstm_12/lstm_cell_24/bias&variables/2/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUElstm_13/lstm_cell_25/kernel&variables/5/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUE%lstm_13/lstm_cell_25/recurrent_kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUElstm_13/lstm_cell_25/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE
 
 
 

Q0
R1
S2

0
1
2
3
 

(0
)1
*2

(0
)1
*2
?
3regularization_losses
Tlayer_regularization_losses
Ulayer_metrics
4	variables
Vnon_trainable_variables
5trainable_variables
Wmetrics

Xlayers
 
 
 
 

0
 
 
 
 
 
 
 

+0
,1
-2

+0
,1
-2
?
Bregularization_losses
Ylayer_regularization_losses
Zlayer_metrics
C	variables
[non_trainable_variables
Dtrainable_variables
\metrics

]layers
 
 
 
 

0
 
 
 
 
 
 
4
	^total
	_count
`	variables
a	keras_api
D
	btotal
	ccount
d
_fn_kwargs
e	variables
f	keras_api
D
	gtotal
	hcount
i
_fn_kwargs
j	variables
k	keras_api
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
^0
_1

`	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

b0
c1

e	variables
QO
VARIABLE_VALUEtotal_24keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_24keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUE
 

g0
h1

j	variables
~|
VARIABLE_VALUEAdam/dense_22/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_22/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_23/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_23/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE"Adam/lstm_12/lstm_cell_24/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE,Adam/lstm_12/lstm_cell_24/recurrent_kernel/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE Adam/lstm_12/lstm_cell_24/bias/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE"Adam/lstm_13/lstm_cell_25/kernel/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE,Adam/lstm_13/lstm_cell_25/recurrent_kernel/mBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE Adam/lstm_13/lstm_cell_25/bias/mBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_22/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_22/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_23/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_23/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE"Adam/lstm_12/lstm_cell_24/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE,Adam/lstm_12/lstm_cell_24/recurrent_kernel/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE Adam/lstm_12/lstm_cell_24/bias/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE"Adam/lstm_13/lstm_cell_25/kernel/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE,Adam/lstm_13/lstm_cell_25/recurrent_kernel/vBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE Adam/lstm_13/lstm_cell_25/bias/vBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_lstm_12_inputPlaceholder*+
_output_shapes
:?????????`*
dtype0* 
shape:?????????`
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_lstm_12_inputlstm_12/lstm_cell_24/kernel%lstm_12/lstm_cell_24/recurrent_kernellstm_12/lstm_cell_24/biasdense_22/kerneldense_22/biaslstm_13/lstm_cell_25/kernel%lstm_13/lstm_cell_25/recurrent_kernellstm_13/lstm_cell_25/biasdense_23/kerneldense_23/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????0*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *-
f(R&
$__inference_signature_wrapper_157779
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_22/kernel/Read/ReadVariableOp!dense_22/bias/Read/ReadVariableOp#dense_23/kernel/Read/ReadVariableOp!dense_23/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp/lstm_12/lstm_cell_24/kernel/Read/ReadVariableOp9lstm_12/lstm_cell_24/recurrent_kernel/Read/ReadVariableOp-lstm_12/lstm_cell_24/bias/Read/ReadVariableOp/lstm_13/lstm_cell_25/kernel/Read/ReadVariableOp9lstm_13/lstm_cell_25/recurrent_kernel/Read/ReadVariableOp-lstm_13/lstm_cell_25/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal_2/Read/ReadVariableOpcount_2/Read/ReadVariableOp*Adam/dense_22/kernel/m/Read/ReadVariableOp(Adam/dense_22/bias/m/Read/ReadVariableOp*Adam/dense_23/kernel/m/Read/ReadVariableOp(Adam/dense_23/bias/m/Read/ReadVariableOp6Adam/lstm_12/lstm_cell_24/kernel/m/Read/ReadVariableOp@Adam/lstm_12/lstm_cell_24/recurrent_kernel/m/Read/ReadVariableOp4Adam/lstm_12/lstm_cell_24/bias/m/Read/ReadVariableOp6Adam/lstm_13/lstm_cell_25/kernel/m/Read/ReadVariableOp@Adam/lstm_13/lstm_cell_25/recurrent_kernel/m/Read/ReadVariableOp4Adam/lstm_13/lstm_cell_25/bias/m/Read/ReadVariableOp*Adam/dense_22/kernel/v/Read/ReadVariableOp(Adam/dense_22/bias/v/Read/ReadVariableOp*Adam/dense_23/kernel/v/Read/ReadVariableOp(Adam/dense_23/bias/v/Read/ReadVariableOp6Adam/lstm_12/lstm_cell_24/kernel/v/Read/ReadVariableOp@Adam/lstm_12/lstm_cell_24/recurrent_kernel/v/Read/ReadVariableOp4Adam/lstm_12/lstm_cell_24/bias/v/Read/ReadVariableOp6Adam/lstm_13/lstm_cell_25/kernel/v/Read/ReadVariableOp@Adam/lstm_13/lstm_cell_25/recurrent_kernel/v/Read/ReadVariableOp4Adam/lstm_13/lstm_cell_25/bias/v/Read/ReadVariableOpConst*6
Tin/
-2+	*
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
__inference__traced_save_158543
?

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_22/kerneldense_22/biasdense_23/kerneldense_23/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratelstm_12/lstm_cell_24/kernel%lstm_12/lstm_cell_24/recurrent_kernellstm_12/lstm_cell_24/biaslstm_13/lstm_cell_25/kernel%lstm_13/lstm_cell_25/recurrent_kernellstm_13/lstm_cell_25/biastotalcounttotal_1count_1total_2count_2Adam/dense_22/kernel/mAdam/dense_22/bias/mAdam/dense_23/kernel/mAdam/dense_23/bias/m"Adam/lstm_12/lstm_cell_24/kernel/m,Adam/lstm_12/lstm_cell_24/recurrent_kernel/m Adam/lstm_12/lstm_cell_24/bias/m"Adam/lstm_13/lstm_cell_25/kernel/m,Adam/lstm_13/lstm_cell_25/recurrent_kernel/m Adam/lstm_13/lstm_cell_25/bias/mAdam/dense_22/kernel/vAdam/dense_22/bias/vAdam/dense_23/kernel/vAdam/dense_23/bias/v"Adam/lstm_12/lstm_cell_24/kernel/v,Adam/lstm_12/lstm_cell_24/recurrent_kernel/v Adam/lstm_12/lstm_cell_24/bias/v"Adam/lstm_13/lstm_cell_25/kernel/v,Adam/lstm_13/lstm_cell_25/recurrent_kernel/v Adam/lstm_13/lstm_cell_25/bias/v*5
Tin.
,2**
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
"__inference__traced_restore_158676??

?
?
I__inference_sequential_19_layer_call_and_return_conditional_losses_157721

inputs
lstm_12_157696
lstm_12_157698
lstm_12_157700
dense_22_157703
dense_22_157705
lstm_13_157708
lstm_13_157710
lstm_13_157712
dense_23_157715
dense_23_157717
identity?? dense_22/StatefulPartitionedCall? dense_23/StatefulPartitionedCall?lstm_12/StatefulPartitionedCall?lstm_13/StatefulPartitionedCall?
lstm_12/StatefulPartitionedCallStatefulPartitionedCallinputslstm_12_157696lstm_12_157698lstm_12_157700*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????`?*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_lstm_12_layer_call_and_return_conditional_losses_1574232!
lstm_12/StatefulPartitionedCall?
 dense_22/StatefulPartitionedCallStatefulPartitionedCall(lstm_12/StatefulPartitionedCall:output:0dense_22_157703dense_22_157705*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????`*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_22_layer_call_and_return_conditional_losses_1574642"
 dense_22/StatefulPartitionedCall?
lstm_13/StatefulPartitionedCallStatefulPartitionedCall)dense_22/StatefulPartitionedCall:output:0lstm_13_157708lstm_13_157710lstm_13_157712*
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
GPU 2J 8? *L
fGRE
C__inference_lstm_13_layer_call_and_return_conditional_losses_1575512!
lstm_13/StatefulPartitionedCall?
 dense_23/StatefulPartitionedCallStatefulPartitionedCall(lstm_13/StatefulPartitionedCall:output:0dense_23_157715dense_23_157717*
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
GPU 2J 8? *M
fHRF
D__inference_dense_23_layer_call_and_return_conditional_losses_1575922"
 dense_23/StatefulPartitionedCall?
IdentityIdentity)dense_23/StatefulPartitionedCall:output:0!^dense_22/StatefulPartitionedCall!^dense_23/StatefulPartitionedCall ^lstm_12/StatefulPartitionedCall ^lstm_13/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????02

Identity"
identityIdentity:output:0*R
_input_shapesA
?:?????????`::::::::::2D
 dense_22/StatefulPartitionedCall dense_22/StatefulPartitionedCall2D
 dense_23/StatefulPartitionedCall dense_23/StatefulPartitionedCall2B
lstm_12/StatefulPartitionedCalllstm_12/StatefulPartitionedCall2B
lstm_13/StatefulPartitionedCalllstm_13/StatefulPartitionedCall:S O
+
_output_shapes
:?????????`
 
_user_specified_nameinputs
?!
?
C__inference_lstm_12_layer_call_and_return_conditional_losses_158057
inputs_0#
mlclstm_readvariableop_resource%
!mlclstm_readvariableop_1_resource%
!mlclstm_readvariableop_2_resource
identity??MLCLSTM/ReadVariableOp?MLCLSTM/ReadVariableOp_1?MLCLSTM/ReadVariableOp_2F
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
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????2	
zeros_1m
MLCLSTM/hidden_sizeConst*
_output_shapes
: *
dtype0*
value
B :?2
MLCLSTM/hidden_sizem
MLCLSTM/output_sizeConst*
_output_shapes
: *
dtype0*
value
B :?2
MLCLSTM/output_size?
MLCLSTM/ReadVariableOpReadVariableOpmlclstm_readvariableop_resource*
_output_shapes
:	?*
dtype02
MLCLSTM/ReadVariableOp?
MLCLSTM/ReadVariableOp_1ReadVariableOp!mlclstm_readvariableop_1_resource* 
_output_shapes
:
??*
dtype02
MLCLSTM/ReadVariableOp_1?
MLCLSTM/ReadVariableOp_2ReadVariableOp!mlclstm_readvariableop_2_resource*
_output_shapes	
:?*
dtype02
MLCLSTM/ReadVariableOp_2?
MLCLSTMMLCLSTMinputs_0MLCLSTM/hidden_size:output:0MLCLSTM/output_size:output:0MLCLSTM/ReadVariableOp:value:0 MLCLSTM/ReadVariableOp_1:value:0 MLCLSTM/ReadVariableOp_2:value:0*
T0*5
_output_shapes#
!:???????????????????*
dropout%    *
return_sequences(2	
MLCLSTM?
IdentityIdentityMLCLSTM:output:0^MLCLSTM/ReadVariableOp^MLCLSTM/ReadVariableOp_1^MLCLSTM/ReadVariableOp_2*
T0*5
_output_shapes#
!:???????????????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????:::20
MLCLSTM/ReadVariableOpMLCLSTM/ReadVariableOp24
MLCLSTM/ReadVariableOp_1MLCLSTM/ReadVariableOp_124
MLCLSTM/ReadVariableOp_2MLCLSTM/ReadVariableOp_2:^ Z
4
_output_shapes"
 :??????????????????
"
_user_specified_name
inputs/0
?
?
(__inference_lstm_13_layer_call_fn_158366

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
GPU 2J 8? *L
fGRE
C__inference_lstm_13_layer_call_and_return_conditional_losses_1575152
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????`:::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????`
 
_user_specified_nameinputs
?`
?
I__inference_sequential_19_layer_call_and_return_conditional_losses_157939

inputs+
'lstm_12_mlclstm_readvariableop_resource-
)lstm_12_mlclstm_readvariableop_1_resource-
)lstm_12_mlclstm_readvariableop_2_resource.
*dense_22_mlcmatmul_readvariableop_resource,
(dense_22_biasadd_readvariableop_resource+
'lstm_13_mlclstm_readvariableop_resource-
)lstm_13_mlclstm_readvariableop_1_resource-
)lstm_13_mlclstm_readvariableop_2_resource.
*dense_23_mlcmatmul_readvariableop_resource,
(dense_23_biasadd_readvariableop_resource
identity??dense_22/BiasAdd/ReadVariableOp?!dense_22/MLCMatMul/ReadVariableOp?dense_23/BiasAdd/ReadVariableOp?!dense_23/MLCMatMul/ReadVariableOp?lstm_12/MLCLSTM/ReadVariableOp? lstm_12/MLCLSTM/ReadVariableOp_1? lstm_12/MLCLSTM/ReadVariableOp_2?lstm_13/MLCLSTM/ReadVariableOp? lstm_13/MLCLSTM/ReadVariableOp_1? lstm_13/MLCLSTM/ReadVariableOp_2T
lstm_12/ShapeShapeinputs*
T0*
_output_shapes
:2
lstm_12/Shape?
lstm_12/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_12/strided_slice/stack?
lstm_12/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_12/strided_slice/stack_1?
lstm_12/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_12/strided_slice/stack_2?
lstm_12/strided_sliceStridedSlicelstm_12/Shape:output:0$lstm_12/strided_slice/stack:output:0&lstm_12/strided_slice/stack_1:output:0&lstm_12/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_12/strided_slicem
lstm_12/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_12/zeros/mul/y?
lstm_12/zeros/mulMullstm_12/strided_slice:output:0lstm_12/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_12/zeros/mulo
lstm_12/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_12/zeros/Less/y?
lstm_12/zeros/LessLesslstm_12/zeros/mul:z:0lstm_12/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_12/zeros/Lesss
lstm_12/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
lstm_12/zeros/packed/1?
lstm_12/zeros/packedPacklstm_12/strided_slice:output:0lstm_12/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_12/zeros/packedo
lstm_12/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_12/zeros/Const?
lstm_12/zerosFilllstm_12/zeros/packed:output:0lstm_12/zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_12/zerosq
lstm_12/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_12/zeros_1/mul/y?
lstm_12/zeros_1/mulMullstm_12/strided_slice:output:0lstm_12/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_12/zeros_1/muls
lstm_12/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_12/zeros_1/Less/y?
lstm_12/zeros_1/LessLesslstm_12/zeros_1/mul:z:0lstm_12/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_12/zeros_1/Lessw
lstm_12/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
lstm_12/zeros_1/packed/1?
lstm_12/zeros_1/packedPacklstm_12/strided_slice:output:0!lstm_12/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_12/zeros_1/packeds
lstm_12/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_12/zeros_1/Const?
lstm_12/zeros_1Filllstm_12/zeros_1/packed:output:0lstm_12/zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_12/zeros_1}
lstm_12/MLCLSTM/hidden_sizeConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_12/MLCLSTM/hidden_size}
lstm_12/MLCLSTM/output_sizeConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_12/MLCLSTM/output_size?
lstm_12/MLCLSTM/ReadVariableOpReadVariableOp'lstm_12_mlclstm_readvariableop_resource*
_output_shapes
:	?*
dtype02 
lstm_12/MLCLSTM/ReadVariableOp?
 lstm_12/MLCLSTM/ReadVariableOp_1ReadVariableOp)lstm_12_mlclstm_readvariableop_1_resource* 
_output_shapes
:
??*
dtype02"
 lstm_12/MLCLSTM/ReadVariableOp_1?
 lstm_12/MLCLSTM/ReadVariableOp_2ReadVariableOp)lstm_12_mlclstm_readvariableop_2_resource*
_output_shapes	
:?*
dtype02"
 lstm_12/MLCLSTM/ReadVariableOp_2?
lstm_12/MLCLSTMMLCLSTMinputs$lstm_12/MLCLSTM/hidden_size:output:0$lstm_12/MLCLSTM/output_size:output:0&lstm_12/MLCLSTM/ReadVariableOp:value:0(lstm_12/MLCLSTM/ReadVariableOp_1:value:0(lstm_12/MLCLSTM/ReadVariableOp_2:value:0*
T0*,
_output_shapes
:?????????`?*
dropout%    *
return_sequences(2
lstm_12/MLCLSTM?
!dense_22/MLCMatMul/ReadVariableOpReadVariableOp*dense_22_mlcmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02#
!dense_22/MLCMatMul/ReadVariableOp?
dense_22/MLCMatMul	MLCMatMullstm_12/MLCLSTM:output:0)dense_22/MLCMatMul/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????`*

input_rank2
dense_22/MLCMatMul?
dense_22/BiasAdd/ReadVariableOpReadVariableOp(dense_22_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_22/BiasAdd/ReadVariableOp?
dense_22/BiasAddBiasAdddense_22/MLCMatMul:product:0'dense_22/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????`2
dense_22/BiasAddw
dense_22/TanhTanhdense_22/BiasAdd:output:0*
T0*+
_output_shapes
:?????????`2
dense_22/Tanh_
lstm_13/ShapeShapedense_22/Tanh:y:0*
T0*
_output_shapes
:2
lstm_13/Shape?
lstm_13/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_13/strided_slice/stack?
lstm_13/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_13/strided_slice/stack_1?
lstm_13/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_13/strided_slice/stack_2?
lstm_13/strided_sliceStridedSlicelstm_13/Shape:output:0$lstm_13/strided_slice/stack:output:0&lstm_13/strided_slice/stack_1:output:0&lstm_13/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_13/strided_slicel
lstm_13/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
lstm_13/zeros/mul/y?
lstm_13/zeros/mulMullstm_13/strided_slice:output:0lstm_13/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_13/zeros/mulo
lstm_13/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_13/zeros/Less/y?
lstm_13/zeros/LessLesslstm_13/zeros/mul:z:0lstm_13/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_13/zeros/Lessr
lstm_13/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
lstm_13/zeros/packed/1?
lstm_13/zeros/packedPacklstm_13/strided_slice:output:0lstm_13/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_13/zeros/packedo
lstm_13/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_13/zeros/Const?
lstm_13/zerosFilllstm_13/zeros/packed:output:0lstm_13/zeros/Const:output:0*
T0*'
_output_shapes
:?????????22
lstm_13/zerosp
lstm_13/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
lstm_13/zeros_1/mul/y?
lstm_13/zeros_1/mulMullstm_13/strided_slice:output:0lstm_13/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_13/zeros_1/muls
lstm_13/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_13/zeros_1/Less/y?
lstm_13/zeros_1/LessLesslstm_13/zeros_1/mul:z:0lstm_13/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_13/zeros_1/Lessv
lstm_13/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
lstm_13/zeros_1/packed/1?
lstm_13/zeros_1/packedPacklstm_13/strided_slice:output:0!lstm_13/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_13/zeros_1/packeds
lstm_13/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_13/zeros_1/Const?
lstm_13/zeros_1Filllstm_13/zeros_1/packed:output:0lstm_13/zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????22
lstm_13/zeros_1|
lstm_13/MLCLSTM/hidden_sizeConst*
_output_shapes
: *
dtype0*
value	B :22
lstm_13/MLCLSTM/hidden_size|
lstm_13/MLCLSTM/output_sizeConst*
_output_shapes
: *
dtype0*
value	B :22
lstm_13/MLCLSTM/output_size?
lstm_13/MLCLSTM/ReadVariableOpReadVariableOp'lstm_13_mlclstm_readvariableop_resource*
_output_shapes
:	?*
dtype02 
lstm_13/MLCLSTM/ReadVariableOp?
 lstm_13/MLCLSTM/ReadVariableOp_1ReadVariableOp)lstm_13_mlclstm_readvariableop_1_resource*
_output_shapes
:	2?*
dtype02"
 lstm_13/MLCLSTM/ReadVariableOp_1?
 lstm_13/MLCLSTM/ReadVariableOp_2ReadVariableOp)lstm_13_mlclstm_readvariableop_2_resource*
_output_shapes	
:?*
dtype02"
 lstm_13/MLCLSTM/ReadVariableOp_2?
lstm_13/MLCLSTMMLCLSTMdense_22/Tanh:y:0$lstm_13/MLCLSTM/hidden_size:output:0$lstm_13/MLCLSTM/output_size:output:0&lstm_13/MLCLSTM/ReadVariableOp:value:0(lstm_13/MLCLSTM/ReadVariableOp_1:value:0(lstm_13/MLCLSTM/ReadVariableOp_2:value:0*
T0*+
_output_shapes
:?????????2*
dropout%    2
lstm_13/MLCLSTM
lstm_13/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   2
lstm_13/Reshape/shape?
lstm_13/ReshapeReshapelstm_13/MLCLSTM:output:0lstm_13/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????22
lstm_13/Reshape?
!dense_23/MLCMatMul/ReadVariableOpReadVariableOp*dense_23_mlcmatmul_readvariableop_resource*
_output_shapes

:20*
dtype02#
!dense_23/MLCMatMul/ReadVariableOp?
dense_23/MLCMatMul	MLCMatMullstm_13/Reshape:output:0)dense_23/MLCMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????02
dense_23/MLCMatMul?
dense_23/BiasAdd/ReadVariableOpReadVariableOp(dense_23_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02!
dense_23/BiasAdd/ReadVariableOp?
dense_23/BiasAddBiasAdddense_23/MLCMatMul:product:0'dense_23/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????02
dense_23/BiasAdds
dense_23/TanhTanhdense_23/BiasAdd:output:0*
T0*'
_output_shapes
:?????????02
dense_23/Tanh?
IdentityIdentitydense_23/Tanh:y:0 ^dense_22/BiasAdd/ReadVariableOp"^dense_22/MLCMatMul/ReadVariableOp ^dense_23/BiasAdd/ReadVariableOp"^dense_23/MLCMatMul/ReadVariableOp^lstm_12/MLCLSTM/ReadVariableOp!^lstm_12/MLCLSTM/ReadVariableOp_1!^lstm_12/MLCLSTM/ReadVariableOp_2^lstm_13/MLCLSTM/ReadVariableOp!^lstm_13/MLCLSTM/ReadVariableOp_1!^lstm_13/MLCLSTM/ReadVariableOp_2*
T0*'
_output_shapes
:?????????02

Identity"
identityIdentity:output:0*R
_input_shapesA
?:?????????`::::::::::2B
dense_22/BiasAdd/ReadVariableOpdense_22/BiasAdd/ReadVariableOp2F
!dense_22/MLCMatMul/ReadVariableOp!dense_22/MLCMatMul/ReadVariableOp2B
dense_23/BiasAdd/ReadVariableOpdense_23/BiasAdd/ReadVariableOp2F
!dense_23/MLCMatMul/ReadVariableOp!dense_23/MLCMatMul/ReadVariableOp2@
lstm_12/MLCLSTM/ReadVariableOplstm_12/MLCLSTM/ReadVariableOp2D
 lstm_12/MLCLSTM/ReadVariableOp_1 lstm_12/MLCLSTM/ReadVariableOp_12D
 lstm_12/MLCLSTM/ReadVariableOp_2 lstm_12/MLCLSTM/ReadVariableOp_22@
lstm_13/MLCLSTM/ReadVariableOplstm_13/MLCLSTM/ReadVariableOp2D
 lstm_13/MLCLSTM/ReadVariableOp_1 lstm_13/MLCLSTM/ReadVariableOp_12D
 lstm_13/MLCLSTM/ReadVariableOp_2 lstm_13/MLCLSTM/ReadVariableOp_2:S O
+
_output_shapes
:?????????`
 
_user_specified_nameinputs
?!
?
C__inference_lstm_12_layer_call_and_return_conditional_losses_157175

inputs#
mlclstm_readvariableop_resource%
!mlclstm_readvariableop_1_resource%
!mlclstm_readvariableop_2_resource
identity??MLCLSTM/ReadVariableOp?MLCLSTM/ReadVariableOp_1?MLCLSTM/ReadVariableOp_2D
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
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????2	
zeros_1m
MLCLSTM/hidden_sizeConst*
_output_shapes
: *
dtype0*
value
B :?2
MLCLSTM/hidden_sizem
MLCLSTM/output_sizeConst*
_output_shapes
: *
dtype0*
value
B :?2
MLCLSTM/output_size?
MLCLSTM/ReadVariableOpReadVariableOpmlclstm_readvariableop_resource*
_output_shapes
:	?*
dtype02
MLCLSTM/ReadVariableOp?
MLCLSTM/ReadVariableOp_1ReadVariableOp!mlclstm_readvariableop_1_resource* 
_output_shapes
:
??*
dtype02
MLCLSTM/ReadVariableOp_1?
MLCLSTM/ReadVariableOp_2ReadVariableOp!mlclstm_readvariableop_2_resource*
_output_shapes	
:?*
dtype02
MLCLSTM/ReadVariableOp_2?
MLCLSTMMLCLSTMinputsMLCLSTM/hidden_size:output:0MLCLSTM/output_size:output:0MLCLSTM/ReadVariableOp:value:0 MLCLSTM/ReadVariableOp_1:value:0 MLCLSTM/ReadVariableOp_2:value:0*
T0*5
_output_shapes#
!:???????????????????*
dropout%    *
return_sequences(2	
MLCLSTM?
IdentityIdentityMLCLSTM:output:0^MLCLSTM/ReadVariableOp^MLCLSTM/ReadVariableOp_1^MLCLSTM/ReadVariableOp_2*
T0*5
_output_shapes#
!:???????????????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????:::20
MLCLSTM/ReadVariableOpMLCLSTM/ReadVariableOp24
MLCLSTM/ReadVariableOp_1MLCLSTM/ReadVariableOp_124
MLCLSTM/ReadVariableOp_2MLCLSTM/ReadVariableOp_2:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?"
?
C__inference_lstm_13_layer_call_and_return_conditional_losses_157515

inputs#
mlclstm_readvariableop_resource%
!mlclstm_readvariableop_1_resource%
!mlclstm_readvariableop_2_resource
identity??MLCLSTM/ReadVariableOp?MLCLSTM/ReadVariableOp_1?MLCLSTM/ReadVariableOp_2D
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
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????22	
zeros_1l
MLCLSTM/hidden_sizeConst*
_output_shapes
: *
dtype0*
value	B :22
MLCLSTM/hidden_sizel
MLCLSTM/output_sizeConst*
_output_shapes
: *
dtype0*
value	B :22
MLCLSTM/output_size?
MLCLSTM/ReadVariableOpReadVariableOpmlclstm_readvariableop_resource*
_output_shapes
:	?*
dtype02
MLCLSTM/ReadVariableOp?
MLCLSTM/ReadVariableOp_1ReadVariableOp!mlclstm_readvariableop_1_resource*
_output_shapes
:	2?*
dtype02
MLCLSTM/ReadVariableOp_1?
MLCLSTM/ReadVariableOp_2ReadVariableOp!mlclstm_readvariableop_2_resource*
_output_shapes	
:?*
dtype02
MLCLSTM/ReadVariableOp_2?
MLCLSTMMLCLSTMinputsMLCLSTM/hidden_size:output:0MLCLSTM/output_size:output:0MLCLSTM/ReadVariableOp:value:0 MLCLSTM/ReadVariableOp_1:value:0 MLCLSTM/ReadVariableOp_2:value:0*
T0*+
_output_shapes
:?????????2*
dropout%    2	
MLCLSTMo
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   2
Reshape/shapey
ReshapeReshapeMLCLSTM:output:0Reshape/shape:output:0*
T0*'
_output_shapes
:?????????22	
Reshape?
IdentityIdentityReshape:output:0^MLCLSTM/ReadVariableOp^MLCLSTM/ReadVariableOp_1^MLCLSTM/ReadVariableOp_2*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????`:::20
MLCLSTM/ReadVariableOpMLCLSTM/ReadVariableOp24
MLCLSTM/ReadVariableOp_1MLCLSTM/ReadVariableOp_124
MLCLSTM/ReadVariableOp_2MLCLSTM/ReadVariableOp_2:S O
+
_output_shapes
:?????????`
 
_user_specified_nameinputs
?
?
(__inference_lstm_12_layer_call_fn_158158

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
:?????????`?*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_lstm_12_layer_call_and_return_conditional_losses_1573892
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:?????????`?2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????`:::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????`
 
_user_specified_nameinputs
?!
?
C__inference_lstm_12_layer_call_and_return_conditional_losses_158023
inputs_0#
mlclstm_readvariableop_resource%
!mlclstm_readvariableop_1_resource%
!mlclstm_readvariableop_2_resource
identity??MLCLSTM/ReadVariableOp?MLCLSTM/ReadVariableOp_1?MLCLSTM/ReadVariableOp_2F
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
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????2	
zeros_1m
MLCLSTM/hidden_sizeConst*
_output_shapes
: *
dtype0*
value
B :?2
MLCLSTM/hidden_sizem
MLCLSTM/output_sizeConst*
_output_shapes
: *
dtype0*
value
B :?2
MLCLSTM/output_size?
MLCLSTM/ReadVariableOpReadVariableOpmlclstm_readvariableop_resource*
_output_shapes
:	?*
dtype02
MLCLSTM/ReadVariableOp?
MLCLSTM/ReadVariableOp_1ReadVariableOp!mlclstm_readvariableop_1_resource* 
_output_shapes
:
??*
dtype02
MLCLSTM/ReadVariableOp_1?
MLCLSTM/ReadVariableOp_2ReadVariableOp!mlclstm_readvariableop_2_resource*
_output_shapes	
:?*
dtype02
MLCLSTM/ReadVariableOp_2?
MLCLSTMMLCLSTMinputs_0MLCLSTM/hidden_size:output:0MLCLSTM/output_size:output:0MLCLSTM/ReadVariableOp:value:0 MLCLSTM/ReadVariableOp_1:value:0 MLCLSTM/ReadVariableOp_2:value:0*
T0*5
_output_shapes#
!:???????????????????*
dropout%    *
return_sequences(2	
MLCLSTM?
IdentityIdentityMLCLSTM:output:0^MLCLSTM/ReadVariableOp^MLCLSTM/ReadVariableOp_1^MLCLSTM/ReadVariableOp_2*
T0*5
_output_shapes#
!:???????????????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????:::20
MLCLSTM/ReadVariableOpMLCLSTM/ReadVariableOp24
MLCLSTM/ReadVariableOp_1MLCLSTM/ReadVariableOp_124
MLCLSTM/ReadVariableOp_2MLCLSTM/ReadVariableOp_2:^ Z
4
_output_shapes"
 :??????????????????
"
_user_specified_name
inputs/0
?"
?
C__inference_lstm_13_layer_call_and_return_conditional_losses_158225
inputs_0#
mlclstm_readvariableop_resource%
!mlclstm_readvariableop_1_resource%
!mlclstm_readvariableop_2_resource
identity??MLCLSTM/ReadVariableOp?MLCLSTM/ReadVariableOp_1?MLCLSTM/ReadVariableOp_2F
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
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????22	
zeros_1l
MLCLSTM/hidden_sizeConst*
_output_shapes
: *
dtype0*
value	B :22
MLCLSTM/hidden_sizel
MLCLSTM/output_sizeConst*
_output_shapes
: *
dtype0*
value	B :22
MLCLSTM/output_size?
MLCLSTM/ReadVariableOpReadVariableOpmlclstm_readvariableop_resource*
_output_shapes
:	?*
dtype02
MLCLSTM/ReadVariableOp?
MLCLSTM/ReadVariableOp_1ReadVariableOp!mlclstm_readvariableop_1_resource*
_output_shapes
:	2?*
dtype02
MLCLSTM/ReadVariableOp_1?
MLCLSTM/ReadVariableOp_2ReadVariableOp!mlclstm_readvariableop_2_resource*
_output_shapes	
:?*
dtype02
MLCLSTM/ReadVariableOp_2?
MLCLSTMMLCLSTMinputs_0MLCLSTM/hidden_size:output:0MLCLSTM/output_size:output:0MLCLSTM/ReadVariableOp:value:0 MLCLSTM/ReadVariableOp_1:value:0 MLCLSTM/ReadVariableOp_2:value:0*
T0*+
_output_shapes
:?????????2*
dropout%    2	
MLCLSTMo
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   2
Reshape/shapey
ReshapeReshapeMLCLSTM:output:0Reshape/shape:output:0*
T0*'
_output_shapes
:?????????22	
Reshape?
IdentityIdentityReshape:output:0^MLCLSTM/ReadVariableOp^MLCLSTM/ReadVariableOp_1^MLCLSTM/ReadVariableOp_2*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????:::20
MLCLSTM/ReadVariableOpMLCLSTM/ReadVariableOp24
MLCLSTM/ReadVariableOp_1MLCLSTM/ReadVariableOp_124
MLCLSTM/ReadVariableOp_2MLCLSTM/ReadVariableOp_2:^ Z
4
_output_shapes"
 :??????????????????
"
_user_specified_name
inputs/0
?
~
)__inference_dense_23_layer_call_fn_158397

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
GPU 2J 8? *M
fHRF
D__inference_dense_23_layer_call_and_return_conditional_losses_1575922
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????02

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????2::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs
? 
?
C__inference_lstm_12_layer_call_and_return_conditional_losses_158113

inputs#
mlclstm_readvariableop_resource%
!mlclstm_readvariableop_1_resource%
!mlclstm_readvariableop_2_resource
identity??MLCLSTM/ReadVariableOp?MLCLSTM/ReadVariableOp_1?MLCLSTM/ReadVariableOp_2D
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
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????2	
zeros_1m
MLCLSTM/hidden_sizeConst*
_output_shapes
: *
dtype0*
value
B :?2
MLCLSTM/hidden_sizem
MLCLSTM/output_sizeConst*
_output_shapes
: *
dtype0*
value
B :?2
MLCLSTM/output_size?
MLCLSTM/ReadVariableOpReadVariableOpmlclstm_readvariableop_resource*
_output_shapes
:	?*
dtype02
MLCLSTM/ReadVariableOp?
MLCLSTM/ReadVariableOp_1ReadVariableOp!mlclstm_readvariableop_1_resource* 
_output_shapes
:
??*
dtype02
MLCLSTM/ReadVariableOp_1?
MLCLSTM/ReadVariableOp_2ReadVariableOp!mlclstm_readvariableop_2_resource*
_output_shapes	
:?*
dtype02
MLCLSTM/ReadVariableOp_2?
MLCLSTMMLCLSTMinputsMLCLSTM/hidden_size:output:0MLCLSTM/output_size:output:0MLCLSTM/ReadVariableOp:value:0 MLCLSTM/ReadVariableOp_1:value:0 MLCLSTM/ReadVariableOp_2:value:0*
T0*,
_output_shapes
:?????????`?*
dropout%    *
return_sequences(2	
MLCLSTM?
IdentityIdentityMLCLSTM:output:0^MLCLSTM/ReadVariableOp^MLCLSTM/ReadVariableOp_1^MLCLSTM/ReadVariableOp_2*
T0*,
_output_shapes
:?????????`?2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????`:::20
MLCLSTM/ReadVariableOpMLCLSTM/ReadVariableOp24
MLCLSTM/ReadVariableOp_1MLCLSTM/ReadVariableOp_124
MLCLSTM/ReadVariableOp_2MLCLSTM/ReadVariableOp_2:S O
+
_output_shapes
:?????????`
 
_user_specified_nameinputs
?|
?	
!__inference__wrapped_model_157024
lstm_12_input9
5sequential_19_lstm_12_mlclstm_readvariableop_resource;
7sequential_19_lstm_12_mlclstm_readvariableop_1_resource;
7sequential_19_lstm_12_mlclstm_readvariableop_2_resource<
8sequential_19_dense_22_mlcmatmul_readvariableop_resource:
6sequential_19_dense_22_biasadd_readvariableop_resource9
5sequential_19_lstm_13_mlclstm_readvariableop_resource;
7sequential_19_lstm_13_mlclstm_readvariableop_1_resource;
7sequential_19_lstm_13_mlclstm_readvariableop_2_resource<
8sequential_19_dense_23_mlcmatmul_readvariableop_resource:
6sequential_19_dense_23_biasadd_readvariableop_resource
identity??-sequential_19/dense_22/BiasAdd/ReadVariableOp?/sequential_19/dense_22/MLCMatMul/ReadVariableOp?-sequential_19/dense_23/BiasAdd/ReadVariableOp?/sequential_19/dense_23/MLCMatMul/ReadVariableOp?,sequential_19/lstm_12/MLCLSTM/ReadVariableOp?.sequential_19/lstm_12/MLCLSTM/ReadVariableOp_1?.sequential_19/lstm_12/MLCLSTM/ReadVariableOp_2?,sequential_19/lstm_13/MLCLSTM/ReadVariableOp?.sequential_19/lstm_13/MLCLSTM/ReadVariableOp_1?.sequential_19/lstm_13/MLCLSTM/ReadVariableOp_2w
sequential_19/lstm_12/ShapeShapelstm_12_input*
T0*
_output_shapes
:2
sequential_19/lstm_12/Shape?
)sequential_19/lstm_12/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)sequential_19/lstm_12/strided_slice/stack?
+sequential_19/lstm_12/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_19/lstm_12/strided_slice/stack_1?
+sequential_19/lstm_12/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_19/lstm_12/strided_slice/stack_2?
#sequential_19/lstm_12/strided_sliceStridedSlice$sequential_19/lstm_12/Shape:output:02sequential_19/lstm_12/strided_slice/stack:output:04sequential_19/lstm_12/strided_slice/stack_1:output:04sequential_19/lstm_12/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#sequential_19/lstm_12/strided_slice?
!sequential_19/lstm_12/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2#
!sequential_19/lstm_12/zeros/mul/y?
sequential_19/lstm_12/zeros/mulMul,sequential_19/lstm_12/strided_slice:output:0*sequential_19/lstm_12/zeros/mul/y:output:0*
T0*
_output_shapes
: 2!
sequential_19/lstm_12/zeros/mul?
"sequential_19/lstm_12/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2$
"sequential_19/lstm_12/zeros/Less/y?
 sequential_19/lstm_12/zeros/LessLess#sequential_19/lstm_12/zeros/mul:z:0+sequential_19/lstm_12/zeros/Less/y:output:0*
T0*
_output_shapes
: 2"
 sequential_19/lstm_12/zeros/Less?
$sequential_19/lstm_12/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2&
$sequential_19/lstm_12/zeros/packed/1?
"sequential_19/lstm_12/zeros/packedPack,sequential_19/lstm_12/strided_slice:output:0-sequential_19/lstm_12/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2$
"sequential_19/lstm_12/zeros/packed?
!sequential_19/lstm_12/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!sequential_19/lstm_12/zeros/Const?
sequential_19/lstm_12/zerosFill+sequential_19/lstm_12/zeros/packed:output:0*sequential_19/lstm_12/zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
sequential_19/lstm_12/zeros?
#sequential_19/lstm_12/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2%
#sequential_19/lstm_12/zeros_1/mul/y?
!sequential_19/lstm_12/zeros_1/mulMul,sequential_19/lstm_12/strided_slice:output:0,sequential_19/lstm_12/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2#
!sequential_19/lstm_12/zeros_1/mul?
$sequential_19/lstm_12/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2&
$sequential_19/lstm_12/zeros_1/Less/y?
"sequential_19/lstm_12/zeros_1/LessLess%sequential_19/lstm_12/zeros_1/mul:z:0-sequential_19/lstm_12/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2$
"sequential_19/lstm_12/zeros_1/Less?
&sequential_19/lstm_12/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2(
&sequential_19/lstm_12/zeros_1/packed/1?
$sequential_19/lstm_12/zeros_1/packedPack,sequential_19/lstm_12/strided_slice:output:0/sequential_19/lstm_12/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2&
$sequential_19/lstm_12/zeros_1/packed?
#sequential_19/lstm_12/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#sequential_19/lstm_12/zeros_1/Const?
sequential_19/lstm_12/zeros_1Fill-sequential_19/lstm_12/zeros_1/packed:output:0,sequential_19/lstm_12/zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????2
sequential_19/lstm_12/zeros_1?
)sequential_19/lstm_12/MLCLSTM/hidden_sizeConst*
_output_shapes
: *
dtype0*
value
B :?2+
)sequential_19/lstm_12/MLCLSTM/hidden_size?
)sequential_19/lstm_12/MLCLSTM/output_sizeConst*
_output_shapes
: *
dtype0*
value
B :?2+
)sequential_19/lstm_12/MLCLSTM/output_size?
,sequential_19/lstm_12/MLCLSTM/ReadVariableOpReadVariableOp5sequential_19_lstm_12_mlclstm_readvariableop_resource*
_output_shapes
:	?*
dtype02.
,sequential_19/lstm_12/MLCLSTM/ReadVariableOp?
.sequential_19/lstm_12/MLCLSTM/ReadVariableOp_1ReadVariableOp7sequential_19_lstm_12_mlclstm_readvariableop_1_resource* 
_output_shapes
:
??*
dtype020
.sequential_19/lstm_12/MLCLSTM/ReadVariableOp_1?
.sequential_19/lstm_12/MLCLSTM/ReadVariableOp_2ReadVariableOp7sequential_19_lstm_12_mlclstm_readvariableop_2_resource*
_output_shapes	
:?*
dtype020
.sequential_19/lstm_12/MLCLSTM/ReadVariableOp_2?
sequential_19/lstm_12/MLCLSTMMLCLSTMlstm_12_input2sequential_19/lstm_12/MLCLSTM/hidden_size:output:02sequential_19/lstm_12/MLCLSTM/output_size:output:04sequential_19/lstm_12/MLCLSTM/ReadVariableOp:value:06sequential_19/lstm_12/MLCLSTM/ReadVariableOp_1:value:06sequential_19/lstm_12/MLCLSTM/ReadVariableOp_2:value:0*
T0*,
_output_shapes
:?????????`?*
dropout%    *
return_sequences(2
sequential_19/lstm_12/MLCLSTM?
/sequential_19/dense_22/MLCMatMul/ReadVariableOpReadVariableOp8sequential_19_dense_22_mlcmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype021
/sequential_19/dense_22/MLCMatMul/ReadVariableOp?
 sequential_19/dense_22/MLCMatMul	MLCMatMul&sequential_19/lstm_12/MLCLSTM:output:07sequential_19/dense_22/MLCMatMul/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????`*

input_rank2"
 sequential_19/dense_22/MLCMatMul?
-sequential_19/dense_22/BiasAdd/ReadVariableOpReadVariableOp6sequential_19_dense_22_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-sequential_19/dense_22/BiasAdd/ReadVariableOp?
sequential_19/dense_22/BiasAddBiasAdd*sequential_19/dense_22/MLCMatMul:product:05sequential_19/dense_22/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????`2 
sequential_19/dense_22/BiasAdd?
sequential_19/dense_22/TanhTanh'sequential_19/dense_22/BiasAdd:output:0*
T0*+
_output_shapes
:?????????`2
sequential_19/dense_22/Tanh?
sequential_19/lstm_13/ShapeShapesequential_19/dense_22/Tanh:y:0*
T0*
_output_shapes
:2
sequential_19/lstm_13/Shape?
)sequential_19/lstm_13/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)sequential_19/lstm_13/strided_slice/stack?
+sequential_19/lstm_13/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_19/lstm_13/strided_slice/stack_1?
+sequential_19/lstm_13/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_19/lstm_13/strided_slice/stack_2?
#sequential_19/lstm_13/strided_sliceStridedSlice$sequential_19/lstm_13/Shape:output:02sequential_19/lstm_13/strided_slice/stack:output:04sequential_19/lstm_13/strided_slice/stack_1:output:04sequential_19/lstm_13/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#sequential_19/lstm_13/strided_slice?
!sequential_19/lstm_13/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22#
!sequential_19/lstm_13/zeros/mul/y?
sequential_19/lstm_13/zeros/mulMul,sequential_19/lstm_13/strided_slice:output:0*sequential_19/lstm_13/zeros/mul/y:output:0*
T0*
_output_shapes
: 2!
sequential_19/lstm_13/zeros/mul?
"sequential_19/lstm_13/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2$
"sequential_19/lstm_13/zeros/Less/y?
 sequential_19/lstm_13/zeros/LessLess#sequential_19/lstm_13/zeros/mul:z:0+sequential_19/lstm_13/zeros/Less/y:output:0*
T0*
_output_shapes
: 2"
 sequential_19/lstm_13/zeros/Less?
$sequential_19/lstm_13/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22&
$sequential_19/lstm_13/zeros/packed/1?
"sequential_19/lstm_13/zeros/packedPack,sequential_19/lstm_13/strided_slice:output:0-sequential_19/lstm_13/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2$
"sequential_19/lstm_13/zeros/packed?
!sequential_19/lstm_13/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!sequential_19/lstm_13/zeros/Const?
sequential_19/lstm_13/zerosFill+sequential_19/lstm_13/zeros/packed:output:0*sequential_19/lstm_13/zeros/Const:output:0*
T0*'
_output_shapes
:?????????22
sequential_19/lstm_13/zeros?
#sequential_19/lstm_13/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22%
#sequential_19/lstm_13/zeros_1/mul/y?
!sequential_19/lstm_13/zeros_1/mulMul,sequential_19/lstm_13/strided_slice:output:0,sequential_19/lstm_13/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2#
!sequential_19/lstm_13/zeros_1/mul?
$sequential_19/lstm_13/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2&
$sequential_19/lstm_13/zeros_1/Less/y?
"sequential_19/lstm_13/zeros_1/LessLess%sequential_19/lstm_13/zeros_1/mul:z:0-sequential_19/lstm_13/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2$
"sequential_19/lstm_13/zeros_1/Less?
&sequential_19/lstm_13/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22(
&sequential_19/lstm_13/zeros_1/packed/1?
$sequential_19/lstm_13/zeros_1/packedPack,sequential_19/lstm_13/strided_slice:output:0/sequential_19/lstm_13/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2&
$sequential_19/lstm_13/zeros_1/packed?
#sequential_19/lstm_13/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#sequential_19/lstm_13/zeros_1/Const?
sequential_19/lstm_13/zeros_1Fill-sequential_19/lstm_13/zeros_1/packed:output:0,sequential_19/lstm_13/zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????22
sequential_19/lstm_13/zeros_1?
)sequential_19/lstm_13/MLCLSTM/hidden_sizeConst*
_output_shapes
: *
dtype0*
value	B :22+
)sequential_19/lstm_13/MLCLSTM/hidden_size?
)sequential_19/lstm_13/MLCLSTM/output_sizeConst*
_output_shapes
: *
dtype0*
value	B :22+
)sequential_19/lstm_13/MLCLSTM/output_size?
,sequential_19/lstm_13/MLCLSTM/ReadVariableOpReadVariableOp5sequential_19_lstm_13_mlclstm_readvariableop_resource*
_output_shapes
:	?*
dtype02.
,sequential_19/lstm_13/MLCLSTM/ReadVariableOp?
.sequential_19/lstm_13/MLCLSTM/ReadVariableOp_1ReadVariableOp7sequential_19_lstm_13_mlclstm_readvariableop_1_resource*
_output_shapes
:	2?*
dtype020
.sequential_19/lstm_13/MLCLSTM/ReadVariableOp_1?
.sequential_19/lstm_13/MLCLSTM/ReadVariableOp_2ReadVariableOp7sequential_19_lstm_13_mlclstm_readvariableop_2_resource*
_output_shapes	
:?*
dtype020
.sequential_19/lstm_13/MLCLSTM/ReadVariableOp_2?
sequential_19/lstm_13/MLCLSTMMLCLSTMsequential_19/dense_22/Tanh:y:02sequential_19/lstm_13/MLCLSTM/hidden_size:output:02sequential_19/lstm_13/MLCLSTM/output_size:output:04sequential_19/lstm_13/MLCLSTM/ReadVariableOp:value:06sequential_19/lstm_13/MLCLSTM/ReadVariableOp_1:value:06sequential_19/lstm_13/MLCLSTM/ReadVariableOp_2:value:0*
T0*+
_output_shapes
:?????????2*
dropout%    2
sequential_19/lstm_13/MLCLSTM?
#sequential_19/lstm_13/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   2%
#sequential_19/lstm_13/Reshape/shape?
sequential_19/lstm_13/ReshapeReshape&sequential_19/lstm_13/MLCLSTM:output:0,sequential_19/lstm_13/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????22
sequential_19/lstm_13/Reshape?
/sequential_19/dense_23/MLCMatMul/ReadVariableOpReadVariableOp8sequential_19_dense_23_mlcmatmul_readvariableop_resource*
_output_shapes

:20*
dtype021
/sequential_19/dense_23/MLCMatMul/ReadVariableOp?
 sequential_19/dense_23/MLCMatMul	MLCMatMul&sequential_19/lstm_13/Reshape:output:07sequential_19/dense_23/MLCMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????02"
 sequential_19/dense_23/MLCMatMul?
-sequential_19/dense_23/BiasAdd/ReadVariableOpReadVariableOp6sequential_19_dense_23_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02/
-sequential_19/dense_23/BiasAdd/ReadVariableOp?
sequential_19/dense_23/BiasAddBiasAdd*sequential_19/dense_23/MLCMatMul:product:05sequential_19/dense_23/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????02 
sequential_19/dense_23/BiasAdd?
sequential_19/dense_23/TanhTanh'sequential_19/dense_23/BiasAdd:output:0*
T0*'
_output_shapes
:?????????02
sequential_19/dense_23/Tanh?
IdentityIdentitysequential_19/dense_23/Tanh:y:0.^sequential_19/dense_22/BiasAdd/ReadVariableOp0^sequential_19/dense_22/MLCMatMul/ReadVariableOp.^sequential_19/dense_23/BiasAdd/ReadVariableOp0^sequential_19/dense_23/MLCMatMul/ReadVariableOp-^sequential_19/lstm_12/MLCLSTM/ReadVariableOp/^sequential_19/lstm_12/MLCLSTM/ReadVariableOp_1/^sequential_19/lstm_12/MLCLSTM/ReadVariableOp_2-^sequential_19/lstm_13/MLCLSTM/ReadVariableOp/^sequential_19/lstm_13/MLCLSTM/ReadVariableOp_1/^sequential_19/lstm_13/MLCLSTM/ReadVariableOp_2*
T0*'
_output_shapes
:?????????02

Identity"
identityIdentity:output:0*R
_input_shapesA
?:?????????`::::::::::2^
-sequential_19/dense_22/BiasAdd/ReadVariableOp-sequential_19/dense_22/BiasAdd/ReadVariableOp2b
/sequential_19/dense_22/MLCMatMul/ReadVariableOp/sequential_19/dense_22/MLCMatMul/ReadVariableOp2^
-sequential_19/dense_23/BiasAdd/ReadVariableOp-sequential_19/dense_23/BiasAdd/ReadVariableOp2b
/sequential_19/dense_23/MLCMatMul/ReadVariableOp/sequential_19/dense_23/MLCMatMul/ReadVariableOp2\
,sequential_19/lstm_12/MLCLSTM/ReadVariableOp,sequential_19/lstm_12/MLCLSTM/ReadVariableOp2`
.sequential_19/lstm_12/MLCLSTM/ReadVariableOp_1.sequential_19/lstm_12/MLCLSTM/ReadVariableOp_12`
.sequential_19/lstm_12/MLCLSTM/ReadVariableOp_2.sequential_19/lstm_12/MLCLSTM/ReadVariableOp_22\
,sequential_19/lstm_13/MLCLSTM/ReadVariableOp,sequential_19/lstm_13/MLCLSTM/ReadVariableOp2`
.sequential_19/lstm_13/MLCLSTM/ReadVariableOp_1.sequential_19/lstm_13/MLCLSTM/ReadVariableOp_12`
.sequential_19/lstm_13/MLCLSTM/ReadVariableOp_2.sequential_19/lstm_13/MLCLSTM/ReadVariableOp_2:Z V
+
_output_shapes
:?????????`
'
_user_specified_namelstm_12_input
?

?
D__inference_dense_23_layer_call_and_return_conditional_losses_157592

inputs%
!mlcmatmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MLCMatMul/ReadVariableOp?
MLCMatMul/ReadVariableOpReadVariableOp!mlcmatmul_readvariableop_resource*
_output_shapes

:20*
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
:?????????2::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
MLCMatMul/ReadVariableOpMLCMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs
?
?
(__inference_lstm_12_layer_call_fn_158068
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
GPU 2J 8? *L
fGRE
C__inference_lstm_12_layer_call_and_return_conditional_losses_1571302
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
?
(__inference_lstm_13_layer_call_fn_158272
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
GPU 2J 8? *L
fGRE
C__inference_lstm_13_layer_call_and_return_conditional_losses_1572962
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????:::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :??????????????????
"
_user_specified_name
inputs/0
?"
?
C__inference_lstm_13_layer_call_and_return_conditional_losses_158355

inputs#
mlclstm_readvariableop_resource%
!mlclstm_readvariableop_1_resource%
!mlclstm_readvariableop_2_resource
identity??MLCLSTM/ReadVariableOp?MLCLSTM/ReadVariableOp_1?MLCLSTM/ReadVariableOp_2D
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
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????22	
zeros_1l
MLCLSTM/hidden_sizeConst*
_output_shapes
: *
dtype0*
value	B :22
MLCLSTM/hidden_sizel
MLCLSTM/output_sizeConst*
_output_shapes
: *
dtype0*
value	B :22
MLCLSTM/output_size?
MLCLSTM/ReadVariableOpReadVariableOpmlclstm_readvariableop_resource*
_output_shapes
:	?*
dtype02
MLCLSTM/ReadVariableOp?
MLCLSTM/ReadVariableOp_1ReadVariableOp!mlclstm_readvariableop_1_resource*
_output_shapes
:	2?*
dtype02
MLCLSTM/ReadVariableOp_1?
MLCLSTM/ReadVariableOp_2ReadVariableOp!mlclstm_readvariableop_2_resource*
_output_shapes	
:?*
dtype02
MLCLSTM/ReadVariableOp_2?
MLCLSTMMLCLSTMinputsMLCLSTM/hidden_size:output:0MLCLSTM/output_size:output:0MLCLSTM/ReadVariableOp:value:0 MLCLSTM/ReadVariableOp_1:value:0 MLCLSTM/ReadVariableOp_2:value:0*
T0*+
_output_shapes
:?????????2*
dropout%    2	
MLCLSTMo
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   2
Reshape/shapey
ReshapeReshapeMLCLSTM:output:0Reshape/shape:output:0*
T0*'
_output_shapes
:?????????22	
Reshape?
IdentityIdentityReshape:output:0^MLCLSTM/ReadVariableOp^MLCLSTM/ReadVariableOp_1^MLCLSTM/ReadVariableOp_2*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????`:::20
MLCLSTM/ReadVariableOpMLCLSTM/ReadVariableOp24
MLCLSTM/ReadVariableOp_1MLCLSTM/ReadVariableOp_124
MLCLSTM/ReadVariableOp_2MLCLSTM/ReadVariableOp_2:S O
+
_output_shapes
:?????????`
 
_user_specified_nameinputs
?	
?
.__inference_sequential_19_layer_call_fn_157744
lstm_12_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalllstm_12_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????0*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_sequential_19_layer_call_and_return_conditional_losses_1577212
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????02

Identity"
identityIdentity:output:0*R
_input_shapesA
?:?????????`::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Z V
+
_output_shapes
:?????????`
'
_user_specified_namelstm_12_input
?	
?
.__inference_sequential_19_layer_call_fn_157691
lstm_12_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalllstm_12_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????0*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_sequential_19_layer_call_and_return_conditional_losses_1576682
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????02

Identity"
identityIdentity:output:0*R
_input_shapesA
?:?????????`::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Z V
+
_output_shapes
:?????????`
'
_user_specified_namelstm_12_input
?
?
(__inference_lstm_13_layer_call_fn_158377

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
GPU 2J 8? *L
fGRE
C__inference_lstm_13_layer_call_and_return_conditional_losses_1575512
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????`:::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????`
 
_user_specified_nameinputs
?
?
$__inference_signature_wrapper_157779
lstm_12_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalllstm_12_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????0*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? **
f%R#
!__inference__wrapped_model_1570242
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????02

Identity"
identityIdentity:output:0*R
_input_shapesA
?:?????????`::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Z V
+
_output_shapes
:?????????`
'
_user_specified_namelstm_12_input
?
?
.__inference_sequential_19_layer_call_fn_157964

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????0*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_sequential_19_layer_call_and_return_conditional_losses_1576682
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????02

Identity"
identityIdentity:output:0*R
_input_shapesA
?:?????????`::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????`
 
_user_specified_nameinputs
?"
?
C__inference_lstm_13_layer_call_and_return_conditional_losses_157343

inputs#
mlclstm_readvariableop_resource%
!mlclstm_readvariableop_1_resource%
!mlclstm_readvariableop_2_resource
identity??MLCLSTM/ReadVariableOp?MLCLSTM/ReadVariableOp_1?MLCLSTM/ReadVariableOp_2D
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
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????22	
zeros_1l
MLCLSTM/hidden_sizeConst*
_output_shapes
: *
dtype0*
value	B :22
MLCLSTM/hidden_sizel
MLCLSTM/output_sizeConst*
_output_shapes
: *
dtype0*
value	B :22
MLCLSTM/output_size?
MLCLSTM/ReadVariableOpReadVariableOpmlclstm_readvariableop_resource*
_output_shapes
:	?*
dtype02
MLCLSTM/ReadVariableOp?
MLCLSTM/ReadVariableOp_1ReadVariableOp!mlclstm_readvariableop_1_resource*
_output_shapes
:	2?*
dtype02
MLCLSTM/ReadVariableOp_1?
MLCLSTM/ReadVariableOp_2ReadVariableOp!mlclstm_readvariableop_2_resource*
_output_shapes	
:?*
dtype02
MLCLSTM/ReadVariableOp_2?
MLCLSTMMLCLSTMinputsMLCLSTM/hidden_size:output:0MLCLSTM/output_size:output:0MLCLSTM/ReadVariableOp:value:0 MLCLSTM/ReadVariableOp_1:value:0 MLCLSTM/ReadVariableOp_2:value:0*
T0*+
_output_shapes
:?????????2*
dropout%    2	
MLCLSTMo
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   2
Reshape/shapey
ReshapeReshapeMLCLSTM:output:0Reshape/shape:output:0*
T0*'
_output_shapes
:?????????22	
Reshape?
IdentityIdentityReshape:output:0^MLCLSTM/ReadVariableOp^MLCLSTM/ReadVariableOp_1^MLCLSTM/ReadVariableOp_2*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????:::20
MLCLSTM/ReadVariableOpMLCLSTM/ReadVariableOp24
MLCLSTM/ReadVariableOp_1MLCLSTM/ReadVariableOp_124
MLCLSTM/ReadVariableOp_2MLCLSTM/ReadVariableOp_2:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
? 
?
C__inference_lstm_12_layer_call_and_return_conditional_losses_158147

inputs#
mlclstm_readvariableop_resource%
!mlclstm_readvariableop_1_resource%
!mlclstm_readvariableop_2_resource
identity??MLCLSTM/ReadVariableOp?MLCLSTM/ReadVariableOp_1?MLCLSTM/ReadVariableOp_2D
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
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????2	
zeros_1m
MLCLSTM/hidden_sizeConst*
_output_shapes
: *
dtype0*
value
B :?2
MLCLSTM/hidden_sizem
MLCLSTM/output_sizeConst*
_output_shapes
: *
dtype0*
value
B :?2
MLCLSTM/output_size?
MLCLSTM/ReadVariableOpReadVariableOpmlclstm_readvariableop_resource*
_output_shapes
:	?*
dtype02
MLCLSTM/ReadVariableOp?
MLCLSTM/ReadVariableOp_1ReadVariableOp!mlclstm_readvariableop_1_resource* 
_output_shapes
:
??*
dtype02
MLCLSTM/ReadVariableOp_1?
MLCLSTM/ReadVariableOp_2ReadVariableOp!mlclstm_readvariableop_2_resource*
_output_shapes	
:?*
dtype02
MLCLSTM/ReadVariableOp_2?
MLCLSTMMLCLSTMinputsMLCLSTM/hidden_size:output:0MLCLSTM/output_size:output:0MLCLSTM/ReadVariableOp:value:0 MLCLSTM/ReadVariableOp_1:value:0 MLCLSTM/ReadVariableOp_2:value:0*
T0*,
_output_shapes
:?????????`?*
dropout%    *
return_sequences(2	
MLCLSTM?
IdentityIdentityMLCLSTM:output:0^MLCLSTM/ReadVariableOp^MLCLSTM/ReadVariableOp_1^MLCLSTM/ReadVariableOp_2*
T0*,
_output_shapes
:?????????`?2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????`:::20
MLCLSTM/ReadVariableOpMLCLSTM/ReadVariableOp24
MLCLSTM/ReadVariableOp_1MLCLSTM/ReadVariableOp_124
MLCLSTM/ReadVariableOp_2MLCLSTM/ReadVariableOp_2:S O
+
_output_shapes
:?????????`
 
_user_specified_nameinputs
?
?
(__inference_lstm_13_layer_call_fn_158283
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
GPU 2J 8? *L
fGRE
C__inference_lstm_13_layer_call_and_return_conditional_losses_1573432
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????:::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :??????????????????
"
_user_specified_name
inputs/0
?
?
.__inference_sequential_19_layer_call_fn_157989

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????0*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_sequential_19_layer_call_and_return_conditional_losses_1577212
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????02

Identity"
identityIdentity:output:0*R
_input_shapesA
?:?????????`::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????`
 
_user_specified_nameinputs
?"
?
C__inference_lstm_13_layer_call_and_return_conditional_losses_158319

inputs#
mlclstm_readvariableop_resource%
!mlclstm_readvariableop_1_resource%
!mlclstm_readvariableop_2_resource
identity??MLCLSTM/ReadVariableOp?MLCLSTM/ReadVariableOp_1?MLCLSTM/ReadVariableOp_2D
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
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????22	
zeros_1l
MLCLSTM/hidden_sizeConst*
_output_shapes
: *
dtype0*
value	B :22
MLCLSTM/hidden_sizel
MLCLSTM/output_sizeConst*
_output_shapes
: *
dtype0*
value	B :22
MLCLSTM/output_size?
MLCLSTM/ReadVariableOpReadVariableOpmlclstm_readvariableop_resource*
_output_shapes
:	?*
dtype02
MLCLSTM/ReadVariableOp?
MLCLSTM/ReadVariableOp_1ReadVariableOp!mlclstm_readvariableop_1_resource*
_output_shapes
:	2?*
dtype02
MLCLSTM/ReadVariableOp_1?
MLCLSTM/ReadVariableOp_2ReadVariableOp!mlclstm_readvariableop_2_resource*
_output_shapes	
:?*
dtype02
MLCLSTM/ReadVariableOp_2?
MLCLSTMMLCLSTMinputsMLCLSTM/hidden_size:output:0MLCLSTM/output_size:output:0MLCLSTM/ReadVariableOp:value:0 MLCLSTM/ReadVariableOp_1:value:0 MLCLSTM/ReadVariableOp_2:value:0*
T0*+
_output_shapes
:?????????2*
dropout%    2	
MLCLSTMo
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   2
Reshape/shapey
ReshapeReshapeMLCLSTM:output:0Reshape/shape:output:0*
T0*'
_output_shapes
:?????????22	
Reshape?
IdentityIdentityReshape:output:0^MLCLSTM/ReadVariableOp^MLCLSTM/ReadVariableOp_1^MLCLSTM/ReadVariableOp_2*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????`:::20
MLCLSTM/ReadVariableOpMLCLSTM/ReadVariableOp24
MLCLSTM/ReadVariableOp_1MLCLSTM/ReadVariableOp_124
MLCLSTM/ReadVariableOp_2MLCLSTM/ReadVariableOp_2:S O
+
_output_shapes
:?????????`
 
_user_specified_nameinputs
?
?
(__inference_lstm_12_layer_call_fn_158079
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
GPU 2J 8? *L
fGRE
C__inference_lstm_12_layer_call_and_return_conditional_losses_1571752
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
?

?
D__inference_dense_23_layer_call_and_return_conditional_losses_158388

inputs%
!mlcmatmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MLCMatMul/ReadVariableOp?
MLCMatMul/ReadVariableOpReadVariableOp!mlcmatmul_readvariableop_resource*
_output_shapes

:20*
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
:?????????2::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
MLCMatMul/ReadVariableOpMLCMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs
?
?
(__inference_lstm_12_layer_call_fn_158169

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
:?????????`?*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_lstm_12_layer_call_and_return_conditional_losses_1574232
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:?????????`?2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????`:::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????`
 
_user_specified_nameinputs
?"
?
C__inference_lstm_13_layer_call_and_return_conditional_losses_158261
inputs_0#
mlclstm_readvariableop_resource%
!mlclstm_readvariableop_1_resource%
!mlclstm_readvariableop_2_resource
identity??MLCLSTM/ReadVariableOp?MLCLSTM/ReadVariableOp_1?MLCLSTM/ReadVariableOp_2F
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
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????22	
zeros_1l
MLCLSTM/hidden_sizeConst*
_output_shapes
: *
dtype0*
value	B :22
MLCLSTM/hidden_sizel
MLCLSTM/output_sizeConst*
_output_shapes
: *
dtype0*
value	B :22
MLCLSTM/output_size?
MLCLSTM/ReadVariableOpReadVariableOpmlclstm_readvariableop_resource*
_output_shapes
:	?*
dtype02
MLCLSTM/ReadVariableOp?
MLCLSTM/ReadVariableOp_1ReadVariableOp!mlclstm_readvariableop_1_resource*
_output_shapes
:	2?*
dtype02
MLCLSTM/ReadVariableOp_1?
MLCLSTM/ReadVariableOp_2ReadVariableOp!mlclstm_readvariableop_2_resource*
_output_shapes	
:?*
dtype02
MLCLSTM/ReadVariableOp_2?
MLCLSTMMLCLSTMinputs_0MLCLSTM/hidden_size:output:0MLCLSTM/output_size:output:0MLCLSTM/ReadVariableOp:value:0 MLCLSTM/ReadVariableOp_1:value:0 MLCLSTM/ReadVariableOp_2:value:0*
T0*+
_output_shapes
:?????????2*
dropout%    2	
MLCLSTMo
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   2
Reshape/shapey
ReshapeReshapeMLCLSTM:output:0Reshape/shape:output:0*
T0*'
_output_shapes
:?????????22	
Reshape?
IdentityIdentityReshape:output:0^MLCLSTM/ReadVariableOp^MLCLSTM/ReadVariableOp_1^MLCLSTM/ReadVariableOp_2*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????:::20
MLCLSTM/ReadVariableOpMLCLSTM/ReadVariableOp24
MLCLSTM/ReadVariableOp_1MLCLSTM/ReadVariableOp_124
MLCLSTM/ReadVariableOp_2MLCLSTM/ReadVariableOp_2:^ Z
4
_output_shapes"
 :??????????????????
"
_user_specified_name
inputs/0
?"
?
C__inference_lstm_13_layer_call_and_return_conditional_losses_157551

inputs#
mlclstm_readvariableop_resource%
!mlclstm_readvariableop_1_resource%
!mlclstm_readvariableop_2_resource
identity??MLCLSTM/ReadVariableOp?MLCLSTM/ReadVariableOp_1?MLCLSTM/ReadVariableOp_2D
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
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????22	
zeros_1l
MLCLSTM/hidden_sizeConst*
_output_shapes
: *
dtype0*
value	B :22
MLCLSTM/hidden_sizel
MLCLSTM/output_sizeConst*
_output_shapes
: *
dtype0*
value	B :22
MLCLSTM/output_size?
MLCLSTM/ReadVariableOpReadVariableOpmlclstm_readvariableop_resource*
_output_shapes
:	?*
dtype02
MLCLSTM/ReadVariableOp?
MLCLSTM/ReadVariableOp_1ReadVariableOp!mlclstm_readvariableop_1_resource*
_output_shapes
:	2?*
dtype02
MLCLSTM/ReadVariableOp_1?
MLCLSTM/ReadVariableOp_2ReadVariableOp!mlclstm_readvariableop_2_resource*
_output_shapes	
:?*
dtype02
MLCLSTM/ReadVariableOp_2?
MLCLSTMMLCLSTMinputsMLCLSTM/hidden_size:output:0MLCLSTM/output_size:output:0MLCLSTM/ReadVariableOp:value:0 MLCLSTM/ReadVariableOp_1:value:0 MLCLSTM/ReadVariableOp_2:value:0*
T0*+
_output_shapes
:?????????2*
dropout%    2	
MLCLSTMo
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   2
Reshape/shapey
ReshapeReshapeMLCLSTM:output:0Reshape/shape:output:0*
T0*'
_output_shapes
:?????????22	
Reshape?
IdentityIdentityReshape:output:0^MLCLSTM/ReadVariableOp^MLCLSTM/ReadVariableOp_1^MLCLSTM/ReadVariableOp_2*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????`:::20
MLCLSTM/ReadVariableOpMLCLSTM/ReadVariableOp24
MLCLSTM/ReadVariableOp_1MLCLSTM/ReadVariableOp_124
MLCLSTM/ReadVariableOp_2MLCLSTM/ReadVariableOp_2:S O
+
_output_shapes
:?????????`
 
_user_specified_nameinputs
?
~
)__inference_dense_22_layer_call_fn_158189

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
 *+
_output_shapes
:?????????`*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_22_layer_call_and_return_conditional_losses_1574642
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????`2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :?????????`?::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:?????????`?
 
_user_specified_nameinputs
?!
?
C__inference_lstm_12_layer_call_and_return_conditional_losses_157130

inputs#
mlclstm_readvariableop_resource%
!mlclstm_readvariableop_1_resource%
!mlclstm_readvariableop_2_resource
identity??MLCLSTM/ReadVariableOp?MLCLSTM/ReadVariableOp_1?MLCLSTM/ReadVariableOp_2D
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
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????2	
zeros_1m
MLCLSTM/hidden_sizeConst*
_output_shapes
: *
dtype0*
value
B :?2
MLCLSTM/hidden_sizem
MLCLSTM/output_sizeConst*
_output_shapes
: *
dtype0*
value
B :?2
MLCLSTM/output_size?
MLCLSTM/ReadVariableOpReadVariableOpmlclstm_readvariableop_resource*
_output_shapes
:	?*
dtype02
MLCLSTM/ReadVariableOp?
MLCLSTM/ReadVariableOp_1ReadVariableOp!mlclstm_readvariableop_1_resource* 
_output_shapes
:
??*
dtype02
MLCLSTM/ReadVariableOp_1?
MLCLSTM/ReadVariableOp_2ReadVariableOp!mlclstm_readvariableop_2_resource*
_output_shapes	
:?*
dtype02
MLCLSTM/ReadVariableOp_2?
MLCLSTMMLCLSTMinputsMLCLSTM/hidden_size:output:0MLCLSTM/output_size:output:0MLCLSTM/ReadVariableOp:value:0 MLCLSTM/ReadVariableOp_1:value:0 MLCLSTM/ReadVariableOp_2:value:0*
T0*5
_output_shapes#
!:???????????????????*
dropout%    *
return_sequences(2	
MLCLSTM?
IdentityIdentityMLCLSTM:output:0^MLCLSTM/ReadVariableOp^MLCLSTM/ReadVariableOp_1^MLCLSTM/ReadVariableOp_2*
T0*5
_output_shapes#
!:???????????????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????:::20
MLCLSTM/ReadVariableOpMLCLSTM/ReadVariableOp24
MLCLSTM/ReadVariableOp_1MLCLSTM/ReadVariableOp_124
MLCLSTM/ReadVariableOp_2MLCLSTM/ReadVariableOp_2:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
?
I__inference_sequential_19_layer_call_and_return_conditional_losses_157609
lstm_12_input
lstm_12_157446
lstm_12_157448
lstm_12_157450
dense_22_157475
dense_22_157477
lstm_13_157574
lstm_13_157576
lstm_13_157578
dense_23_157603
dense_23_157605
identity?? dense_22/StatefulPartitionedCall? dense_23/StatefulPartitionedCall?lstm_12/StatefulPartitionedCall?lstm_13/StatefulPartitionedCall?
lstm_12/StatefulPartitionedCallStatefulPartitionedCalllstm_12_inputlstm_12_157446lstm_12_157448lstm_12_157450*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????`?*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_lstm_12_layer_call_and_return_conditional_losses_1573892!
lstm_12/StatefulPartitionedCall?
 dense_22/StatefulPartitionedCallStatefulPartitionedCall(lstm_12/StatefulPartitionedCall:output:0dense_22_157475dense_22_157477*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????`*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_22_layer_call_and_return_conditional_losses_1574642"
 dense_22/StatefulPartitionedCall?
lstm_13/StatefulPartitionedCallStatefulPartitionedCall)dense_22/StatefulPartitionedCall:output:0lstm_13_157574lstm_13_157576lstm_13_157578*
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
GPU 2J 8? *L
fGRE
C__inference_lstm_13_layer_call_and_return_conditional_losses_1575152!
lstm_13/StatefulPartitionedCall?
 dense_23/StatefulPartitionedCallStatefulPartitionedCall(lstm_13/StatefulPartitionedCall:output:0dense_23_157603dense_23_157605*
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
GPU 2J 8? *M
fHRF
D__inference_dense_23_layer_call_and_return_conditional_losses_1575922"
 dense_23/StatefulPartitionedCall?
IdentityIdentity)dense_23/StatefulPartitionedCall:output:0!^dense_22/StatefulPartitionedCall!^dense_23/StatefulPartitionedCall ^lstm_12/StatefulPartitionedCall ^lstm_13/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????02

Identity"
identityIdentity:output:0*R
_input_shapesA
?:?????????`::::::::::2D
 dense_22/StatefulPartitionedCall dense_22/StatefulPartitionedCall2D
 dense_23/StatefulPartitionedCall dense_23/StatefulPartitionedCall2B
lstm_12/StatefulPartitionedCalllstm_12/StatefulPartitionedCall2B
lstm_13/StatefulPartitionedCalllstm_13/StatefulPartitionedCall:Z V
+
_output_shapes
:?????????`
'
_user_specified_namelstm_12_input
?

?
D__inference_dense_22_layer_call_and_return_conditional_losses_157464

inputs%
!mlcmatmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MLCMatMul/ReadVariableOp?
MLCMatMul/ReadVariableOpReadVariableOp!mlcmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MLCMatMul/ReadVariableOp?
	MLCMatMul	MLCMatMulinputs MLCMatMul/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????`*

input_rank2
	MLCMatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMLCMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????`2	
BiasAdd\
TanhTanhBiasAdd:output:0*
T0*+
_output_shapes
:?????????`2
Tanh?
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MLCMatMul/ReadVariableOp*
T0*+
_output_shapes
:?????????`2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :?????????`?::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
MLCMatMul/ReadVariableOpMLCMatMul/ReadVariableOp:T P
,
_output_shapes
:?????????`?
 
_user_specified_nameinputs
?
?
I__inference_sequential_19_layer_call_and_return_conditional_losses_157668

inputs
lstm_12_157643
lstm_12_157645
lstm_12_157647
dense_22_157650
dense_22_157652
lstm_13_157655
lstm_13_157657
lstm_13_157659
dense_23_157662
dense_23_157664
identity?? dense_22/StatefulPartitionedCall? dense_23/StatefulPartitionedCall?lstm_12/StatefulPartitionedCall?lstm_13/StatefulPartitionedCall?
lstm_12/StatefulPartitionedCallStatefulPartitionedCallinputslstm_12_157643lstm_12_157645lstm_12_157647*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????`?*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_lstm_12_layer_call_and_return_conditional_losses_1573892!
lstm_12/StatefulPartitionedCall?
 dense_22/StatefulPartitionedCallStatefulPartitionedCall(lstm_12/StatefulPartitionedCall:output:0dense_22_157650dense_22_157652*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????`*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_22_layer_call_and_return_conditional_losses_1574642"
 dense_22/StatefulPartitionedCall?
lstm_13/StatefulPartitionedCallStatefulPartitionedCall)dense_22/StatefulPartitionedCall:output:0lstm_13_157655lstm_13_157657lstm_13_157659*
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
GPU 2J 8? *L
fGRE
C__inference_lstm_13_layer_call_and_return_conditional_losses_1575152!
lstm_13/StatefulPartitionedCall?
 dense_23/StatefulPartitionedCallStatefulPartitionedCall(lstm_13/StatefulPartitionedCall:output:0dense_23_157662dense_23_157664*
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
GPU 2J 8? *M
fHRF
D__inference_dense_23_layer_call_and_return_conditional_losses_1575922"
 dense_23/StatefulPartitionedCall?
IdentityIdentity)dense_23/StatefulPartitionedCall:output:0!^dense_22/StatefulPartitionedCall!^dense_23/StatefulPartitionedCall ^lstm_12/StatefulPartitionedCall ^lstm_13/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????02

Identity"
identityIdentity:output:0*R
_input_shapesA
?:?????????`::::::::::2D
 dense_22/StatefulPartitionedCall dense_22/StatefulPartitionedCall2D
 dense_23/StatefulPartitionedCall dense_23/StatefulPartitionedCall2B
lstm_12/StatefulPartitionedCalllstm_12/StatefulPartitionedCall2B
lstm_13/StatefulPartitionedCalllstm_13/StatefulPartitionedCall:S O
+
_output_shapes
:?????????`
 
_user_specified_nameinputs
??
?
"__inference__traced_restore_158676
file_prefix$
 assignvariableop_dense_22_kernel$
 assignvariableop_1_dense_22_bias&
"assignvariableop_2_dense_23_kernel$
 assignvariableop_3_dense_23_bias 
assignvariableop_4_adam_iter"
assignvariableop_5_adam_beta_1"
assignvariableop_6_adam_beta_2!
assignvariableop_7_adam_decay)
%assignvariableop_8_adam_learning_rate2
.assignvariableop_9_lstm_12_lstm_cell_24_kernel=
9assignvariableop_10_lstm_12_lstm_cell_24_recurrent_kernel1
-assignvariableop_11_lstm_12_lstm_cell_24_bias3
/assignvariableop_12_lstm_13_lstm_cell_25_kernel=
9assignvariableop_13_lstm_13_lstm_cell_25_recurrent_kernel1
-assignvariableop_14_lstm_13_lstm_cell_25_bias
assignvariableop_15_total
assignvariableop_16_count
assignvariableop_17_total_1
assignvariableop_18_count_1
assignvariableop_19_total_2
assignvariableop_20_count_2.
*assignvariableop_21_adam_dense_22_kernel_m,
(assignvariableop_22_adam_dense_22_bias_m.
*assignvariableop_23_adam_dense_23_kernel_m,
(assignvariableop_24_adam_dense_23_bias_m:
6assignvariableop_25_adam_lstm_12_lstm_cell_24_kernel_mD
@assignvariableop_26_adam_lstm_12_lstm_cell_24_recurrent_kernel_m8
4assignvariableop_27_adam_lstm_12_lstm_cell_24_bias_m:
6assignvariableop_28_adam_lstm_13_lstm_cell_25_kernel_mD
@assignvariableop_29_adam_lstm_13_lstm_cell_25_recurrent_kernel_m8
4assignvariableop_30_adam_lstm_13_lstm_cell_25_bias_m.
*assignvariableop_31_adam_dense_22_kernel_v,
(assignvariableop_32_adam_dense_22_bias_v.
*assignvariableop_33_adam_dense_23_kernel_v,
(assignvariableop_34_adam_dense_23_bias_v:
6assignvariableop_35_adam_lstm_12_lstm_cell_24_kernel_vD
@assignvariableop_36_adam_lstm_12_lstm_cell_24_recurrent_kernel_v8
4assignvariableop_37_adam_lstm_12_lstm_cell_24_bias_v:
6assignvariableop_38_adam_lstm_13_lstm_cell_25_kernel_vD
@assignvariableop_39_adam_lstm_13_lstm_cell_25_recurrent_kernel_v8
4assignvariableop_40_adam_lstm_13_lstm_cell_25_bias_v
identity_42??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:**
dtype0*?
value?B?*B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:**
dtype0*g
value^B\*B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::*8
dtypes.
,2*	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp assignvariableop_dense_22_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_22_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp"assignvariableop_2_dense_23_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp assignvariableop_3_dense_23_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_iterIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_beta_1Identity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_beta_2Identity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_decayIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp%assignvariableop_8_adam_learning_rateIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp.assignvariableop_9_lstm_12_lstm_cell_24_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp9assignvariableop_10_lstm_12_lstm_cell_24_recurrent_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp-assignvariableop_11_lstm_12_lstm_cell_24_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp/assignvariableop_12_lstm_13_lstm_cell_25_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp9assignvariableop_13_lstm_13_lstm_cell_25_recurrent_kernelIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp-assignvariableop_14_lstm_13_lstm_cell_25_biasIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOpassignvariableop_15_totalIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOpassignvariableop_16_countIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOpassignvariableop_17_total_1Identity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOpassignvariableop_18_count_1Identity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOpassignvariableop_19_total_2Identity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOpassignvariableop_20_count_2Identity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp*assignvariableop_21_adam_dense_22_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp(assignvariableop_22_adam_dense_22_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp*assignvariableop_23_adam_dense_23_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp(assignvariableop_24_adam_dense_23_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp6assignvariableop_25_adam_lstm_12_lstm_cell_24_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp@assignvariableop_26_adam_lstm_12_lstm_cell_24_recurrent_kernel_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp4assignvariableop_27_adam_lstm_12_lstm_cell_24_bias_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp6assignvariableop_28_adam_lstm_13_lstm_cell_25_kernel_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp@assignvariableop_29_adam_lstm_13_lstm_cell_25_recurrent_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp4assignvariableop_30_adam_lstm_13_lstm_cell_25_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp*assignvariableop_31_adam_dense_22_kernel_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp(assignvariableop_32_adam_dense_22_bias_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOp*assignvariableop_33_adam_dense_23_kernel_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp(assignvariableop_34_adam_dense_23_bias_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOp6assignvariableop_35_adam_lstm_12_lstm_cell_24_kernel_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOp@assignvariableop_36_adam_lstm_12_lstm_cell_24_recurrent_kernel_vIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOp4assignvariableop_37_adam_lstm_12_lstm_cell_24_bias_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOp6assignvariableop_38_adam_lstm_13_lstm_cell_25_kernel_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39?
AssignVariableOp_39AssignVariableOp@assignvariableop_39_adam_lstm_13_lstm_cell_25_recurrent_kernel_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40?
AssignVariableOp_40AssignVariableOp4assignvariableop_40_adam_lstm_13_lstm_cell_25_bias_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_409
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_41Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_41?
Identity_42IdentityIdentity_41:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_42"#
identity_42Identity_42:output:0*?
_input_shapes?
?: :::::::::::::::::::::::::::::::::::::::::2$
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
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402(
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
?
?
I__inference_sequential_19_layer_call_and_return_conditional_losses_157637
lstm_12_input
lstm_12_157612
lstm_12_157614
lstm_12_157616
dense_22_157619
dense_22_157621
lstm_13_157624
lstm_13_157626
lstm_13_157628
dense_23_157631
dense_23_157633
identity?? dense_22/StatefulPartitionedCall? dense_23/StatefulPartitionedCall?lstm_12/StatefulPartitionedCall?lstm_13/StatefulPartitionedCall?
lstm_12/StatefulPartitionedCallStatefulPartitionedCalllstm_12_inputlstm_12_157612lstm_12_157614lstm_12_157616*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????`?*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_lstm_12_layer_call_and_return_conditional_losses_1574232!
lstm_12/StatefulPartitionedCall?
 dense_22/StatefulPartitionedCallStatefulPartitionedCall(lstm_12/StatefulPartitionedCall:output:0dense_22_157619dense_22_157621*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????`*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_22_layer_call_and_return_conditional_losses_1574642"
 dense_22/StatefulPartitionedCall?
lstm_13/StatefulPartitionedCallStatefulPartitionedCall)dense_22/StatefulPartitionedCall:output:0lstm_13_157624lstm_13_157626lstm_13_157628*
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
GPU 2J 8? *L
fGRE
C__inference_lstm_13_layer_call_and_return_conditional_losses_1575512!
lstm_13/StatefulPartitionedCall?
 dense_23/StatefulPartitionedCallStatefulPartitionedCall(lstm_13/StatefulPartitionedCall:output:0dense_23_157631dense_23_157633*
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
GPU 2J 8? *M
fHRF
D__inference_dense_23_layer_call_and_return_conditional_losses_1575922"
 dense_23/StatefulPartitionedCall?
IdentityIdentity)dense_23/StatefulPartitionedCall:output:0!^dense_22/StatefulPartitionedCall!^dense_23/StatefulPartitionedCall ^lstm_12/StatefulPartitionedCall ^lstm_13/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????02

Identity"
identityIdentity:output:0*R
_input_shapesA
?:?????????`::::::::::2D
 dense_22/StatefulPartitionedCall dense_22/StatefulPartitionedCall2D
 dense_23/StatefulPartitionedCall dense_23/StatefulPartitionedCall2B
lstm_12/StatefulPartitionedCalllstm_12/StatefulPartitionedCall2B
lstm_13/StatefulPartitionedCalllstm_13/StatefulPartitionedCall:Z V
+
_output_shapes
:?????????`
'
_user_specified_namelstm_12_input
?`
?
I__inference_sequential_19_layer_call_and_return_conditional_losses_157859

inputs+
'lstm_12_mlclstm_readvariableop_resource-
)lstm_12_mlclstm_readvariableop_1_resource-
)lstm_12_mlclstm_readvariableop_2_resource.
*dense_22_mlcmatmul_readvariableop_resource,
(dense_22_biasadd_readvariableop_resource+
'lstm_13_mlclstm_readvariableop_resource-
)lstm_13_mlclstm_readvariableop_1_resource-
)lstm_13_mlclstm_readvariableop_2_resource.
*dense_23_mlcmatmul_readvariableop_resource,
(dense_23_biasadd_readvariableop_resource
identity??dense_22/BiasAdd/ReadVariableOp?!dense_22/MLCMatMul/ReadVariableOp?dense_23/BiasAdd/ReadVariableOp?!dense_23/MLCMatMul/ReadVariableOp?lstm_12/MLCLSTM/ReadVariableOp? lstm_12/MLCLSTM/ReadVariableOp_1? lstm_12/MLCLSTM/ReadVariableOp_2?lstm_13/MLCLSTM/ReadVariableOp? lstm_13/MLCLSTM/ReadVariableOp_1? lstm_13/MLCLSTM/ReadVariableOp_2T
lstm_12/ShapeShapeinputs*
T0*
_output_shapes
:2
lstm_12/Shape?
lstm_12/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_12/strided_slice/stack?
lstm_12/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_12/strided_slice/stack_1?
lstm_12/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_12/strided_slice/stack_2?
lstm_12/strided_sliceStridedSlicelstm_12/Shape:output:0$lstm_12/strided_slice/stack:output:0&lstm_12/strided_slice/stack_1:output:0&lstm_12/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_12/strided_slicem
lstm_12/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_12/zeros/mul/y?
lstm_12/zeros/mulMullstm_12/strided_slice:output:0lstm_12/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_12/zeros/mulo
lstm_12/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_12/zeros/Less/y?
lstm_12/zeros/LessLesslstm_12/zeros/mul:z:0lstm_12/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_12/zeros/Lesss
lstm_12/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
lstm_12/zeros/packed/1?
lstm_12/zeros/packedPacklstm_12/strided_slice:output:0lstm_12/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_12/zeros/packedo
lstm_12/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_12/zeros/Const?
lstm_12/zerosFilllstm_12/zeros/packed:output:0lstm_12/zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_12/zerosq
lstm_12/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_12/zeros_1/mul/y?
lstm_12/zeros_1/mulMullstm_12/strided_slice:output:0lstm_12/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_12/zeros_1/muls
lstm_12/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_12/zeros_1/Less/y?
lstm_12/zeros_1/LessLesslstm_12/zeros_1/mul:z:0lstm_12/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_12/zeros_1/Lessw
lstm_12/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
lstm_12/zeros_1/packed/1?
lstm_12/zeros_1/packedPacklstm_12/strided_slice:output:0!lstm_12/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_12/zeros_1/packeds
lstm_12/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_12/zeros_1/Const?
lstm_12/zeros_1Filllstm_12/zeros_1/packed:output:0lstm_12/zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_12/zeros_1}
lstm_12/MLCLSTM/hidden_sizeConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_12/MLCLSTM/hidden_size}
lstm_12/MLCLSTM/output_sizeConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_12/MLCLSTM/output_size?
lstm_12/MLCLSTM/ReadVariableOpReadVariableOp'lstm_12_mlclstm_readvariableop_resource*
_output_shapes
:	?*
dtype02 
lstm_12/MLCLSTM/ReadVariableOp?
 lstm_12/MLCLSTM/ReadVariableOp_1ReadVariableOp)lstm_12_mlclstm_readvariableop_1_resource* 
_output_shapes
:
??*
dtype02"
 lstm_12/MLCLSTM/ReadVariableOp_1?
 lstm_12/MLCLSTM/ReadVariableOp_2ReadVariableOp)lstm_12_mlclstm_readvariableop_2_resource*
_output_shapes	
:?*
dtype02"
 lstm_12/MLCLSTM/ReadVariableOp_2?
lstm_12/MLCLSTMMLCLSTMinputs$lstm_12/MLCLSTM/hidden_size:output:0$lstm_12/MLCLSTM/output_size:output:0&lstm_12/MLCLSTM/ReadVariableOp:value:0(lstm_12/MLCLSTM/ReadVariableOp_1:value:0(lstm_12/MLCLSTM/ReadVariableOp_2:value:0*
T0*,
_output_shapes
:?????????`?*
dropout%    *
return_sequences(2
lstm_12/MLCLSTM?
!dense_22/MLCMatMul/ReadVariableOpReadVariableOp*dense_22_mlcmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02#
!dense_22/MLCMatMul/ReadVariableOp?
dense_22/MLCMatMul	MLCMatMullstm_12/MLCLSTM:output:0)dense_22/MLCMatMul/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????`*

input_rank2
dense_22/MLCMatMul?
dense_22/BiasAdd/ReadVariableOpReadVariableOp(dense_22_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_22/BiasAdd/ReadVariableOp?
dense_22/BiasAddBiasAdddense_22/MLCMatMul:product:0'dense_22/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????`2
dense_22/BiasAddw
dense_22/TanhTanhdense_22/BiasAdd:output:0*
T0*+
_output_shapes
:?????????`2
dense_22/Tanh_
lstm_13/ShapeShapedense_22/Tanh:y:0*
T0*
_output_shapes
:2
lstm_13/Shape?
lstm_13/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_13/strided_slice/stack?
lstm_13/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_13/strided_slice/stack_1?
lstm_13/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_13/strided_slice/stack_2?
lstm_13/strided_sliceStridedSlicelstm_13/Shape:output:0$lstm_13/strided_slice/stack:output:0&lstm_13/strided_slice/stack_1:output:0&lstm_13/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_13/strided_slicel
lstm_13/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
lstm_13/zeros/mul/y?
lstm_13/zeros/mulMullstm_13/strided_slice:output:0lstm_13/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_13/zeros/mulo
lstm_13/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_13/zeros/Less/y?
lstm_13/zeros/LessLesslstm_13/zeros/mul:z:0lstm_13/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_13/zeros/Lessr
lstm_13/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
lstm_13/zeros/packed/1?
lstm_13/zeros/packedPacklstm_13/strided_slice:output:0lstm_13/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_13/zeros/packedo
lstm_13/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_13/zeros/Const?
lstm_13/zerosFilllstm_13/zeros/packed:output:0lstm_13/zeros/Const:output:0*
T0*'
_output_shapes
:?????????22
lstm_13/zerosp
lstm_13/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
lstm_13/zeros_1/mul/y?
lstm_13/zeros_1/mulMullstm_13/strided_slice:output:0lstm_13/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_13/zeros_1/muls
lstm_13/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_13/zeros_1/Less/y?
lstm_13/zeros_1/LessLesslstm_13/zeros_1/mul:z:0lstm_13/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_13/zeros_1/Lessv
lstm_13/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
lstm_13/zeros_1/packed/1?
lstm_13/zeros_1/packedPacklstm_13/strided_slice:output:0!lstm_13/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_13/zeros_1/packeds
lstm_13/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_13/zeros_1/Const?
lstm_13/zeros_1Filllstm_13/zeros_1/packed:output:0lstm_13/zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????22
lstm_13/zeros_1|
lstm_13/MLCLSTM/hidden_sizeConst*
_output_shapes
: *
dtype0*
value	B :22
lstm_13/MLCLSTM/hidden_size|
lstm_13/MLCLSTM/output_sizeConst*
_output_shapes
: *
dtype0*
value	B :22
lstm_13/MLCLSTM/output_size?
lstm_13/MLCLSTM/ReadVariableOpReadVariableOp'lstm_13_mlclstm_readvariableop_resource*
_output_shapes
:	?*
dtype02 
lstm_13/MLCLSTM/ReadVariableOp?
 lstm_13/MLCLSTM/ReadVariableOp_1ReadVariableOp)lstm_13_mlclstm_readvariableop_1_resource*
_output_shapes
:	2?*
dtype02"
 lstm_13/MLCLSTM/ReadVariableOp_1?
 lstm_13/MLCLSTM/ReadVariableOp_2ReadVariableOp)lstm_13_mlclstm_readvariableop_2_resource*
_output_shapes	
:?*
dtype02"
 lstm_13/MLCLSTM/ReadVariableOp_2?
lstm_13/MLCLSTMMLCLSTMdense_22/Tanh:y:0$lstm_13/MLCLSTM/hidden_size:output:0$lstm_13/MLCLSTM/output_size:output:0&lstm_13/MLCLSTM/ReadVariableOp:value:0(lstm_13/MLCLSTM/ReadVariableOp_1:value:0(lstm_13/MLCLSTM/ReadVariableOp_2:value:0*
T0*+
_output_shapes
:?????????2*
dropout%    2
lstm_13/MLCLSTM
lstm_13/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   2
lstm_13/Reshape/shape?
lstm_13/ReshapeReshapelstm_13/MLCLSTM:output:0lstm_13/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????22
lstm_13/Reshape?
!dense_23/MLCMatMul/ReadVariableOpReadVariableOp*dense_23_mlcmatmul_readvariableop_resource*
_output_shapes

:20*
dtype02#
!dense_23/MLCMatMul/ReadVariableOp?
dense_23/MLCMatMul	MLCMatMullstm_13/Reshape:output:0)dense_23/MLCMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????02
dense_23/MLCMatMul?
dense_23/BiasAdd/ReadVariableOpReadVariableOp(dense_23_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02!
dense_23/BiasAdd/ReadVariableOp?
dense_23/BiasAddBiasAdddense_23/MLCMatMul:product:0'dense_23/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????02
dense_23/BiasAdds
dense_23/TanhTanhdense_23/BiasAdd:output:0*
T0*'
_output_shapes
:?????????02
dense_23/Tanh?
IdentityIdentitydense_23/Tanh:y:0 ^dense_22/BiasAdd/ReadVariableOp"^dense_22/MLCMatMul/ReadVariableOp ^dense_23/BiasAdd/ReadVariableOp"^dense_23/MLCMatMul/ReadVariableOp^lstm_12/MLCLSTM/ReadVariableOp!^lstm_12/MLCLSTM/ReadVariableOp_1!^lstm_12/MLCLSTM/ReadVariableOp_2^lstm_13/MLCLSTM/ReadVariableOp!^lstm_13/MLCLSTM/ReadVariableOp_1!^lstm_13/MLCLSTM/ReadVariableOp_2*
T0*'
_output_shapes
:?????????02

Identity"
identityIdentity:output:0*R
_input_shapesA
?:?????????`::::::::::2B
dense_22/BiasAdd/ReadVariableOpdense_22/BiasAdd/ReadVariableOp2F
!dense_22/MLCMatMul/ReadVariableOp!dense_22/MLCMatMul/ReadVariableOp2B
dense_23/BiasAdd/ReadVariableOpdense_23/BiasAdd/ReadVariableOp2F
!dense_23/MLCMatMul/ReadVariableOp!dense_23/MLCMatMul/ReadVariableOp2@
lstm_12/MLCLSTM/ReadVariableOplstm_12/MLCLSTM/ReadVariableOp2D
 lstm_12/MLCLSTM/ReadVariableOp_1 lstm_12/MLCLSTM/ReadVariableOp_12D
 lstm_12/MLCLSTM/ReadVariableOp_2 lstm_12/MLCLSTM/ReadVariableOp_22@
lstm_13/MLCLSTM/ReadVariableOplstm_13/MLCLSTM/ReadVariableOp2D
 lstm_13/MLCLSTM/ReadVariableOp_1 lstm_13/MLCLSTM/ReadVariableOp_12D
 lstm_13/MLCLSTM/ReadVariableOp_2 lstm_13/MLCLSTM/ReadVariableOp_2:S O
+
_output_shapes
:?????????`
 
_user_specified_nameinputs
? 
?
C__inference_lstm_12_layer_call_and_return_conditional_losses_157423

inputs#
mlclstm_readvariableop_resource%
!mlclstm_readvariableop_1_resource%
!mlclstm_readvariableop_2_resource
identity??MLCLSTM/ReadVariableOp?MLCLSTM/ReadVariableOp_1?MLCLSTM/ReadVariableOp_2D
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
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????2	
zeros_1m
MLCLSTM/hidden_sizeConst*
_output_shapes
: *
dtype0*
value
B :?2
MLCLSTM/hidden_sizem
MLCLSTM/output_sizeConst*
_output_shapes
: *
dtype0*
value
B :?2
MLCLSTM/output_size?
MLCLSTM/ReadVariableOpReadVariableOpmlclstm_readvariableop_resource*
_output_shapes
:	?*
dtype02
MLCLSTM/ReadVariableOp?
MLCLSTM/ReadVariableOp_1ReadVariableOp!mlclstm_readvariableop_1_resource* 
_output_shapes
:
??*
dtype02
MLCLSTM/ReadVariableOp_1?
MLCLSTM/ReadVariableOp_2ReadVariableOp!mlclstm_readvariableop_2_resource*
_output_shapes	
:?*
dtype02
MLCLSTM/ReadVariableOp_2?
MLCLSTMMLCLSTMinputsMLCLSTM/hidden_size:output:0MLCLSTM/output_size:output:0MLCLSTM/ReadVariableOp:value:0 MLCLSTM/ReadVariableOp_1:value:0 MLCLSTM/ReadVariableOp_2:value:0*
T0*,
_output_shapes
:?????????`?*
dropout%    *
return_sequences(2	
MLCLSTM?
IdentityIdentityMLCLSTM:output:0^MLCLSTM/ReadVariableOp^MLCLSTM/ReadVariableOp_1^MLCLSTM/ReadVariableOp_2*
T0*,
_output_shapes
:?????????`?2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????`:::20
MLCLSTM/ReadVariableOpMLCLSTM/ReadVariableOp24
MLCLSTM/ReadVariableOp_1MLCLSTM/ReadVariableOp_124
MLCLSTM/ReadVariableOp_2MLCLSTM/ReadVariableOp_2:S O
+
_output_shapes
:?????????`
 
_user_specified_nameinputs
?W
?
__inference__traced_save_158543
file_prefix.
*savev2_dense_22_kernel_read_readvariableop,
(savev2_dense_22_bias_read_readvariableop.
*savev2_dense_23_kernel_read_readvariableop,
(savev2_dense_23_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop:
6savev2_lstm_12_lstm_cell_24_kernel_read_readvariableopD
@savev2_lstm_12_lstm_cell_24_recurrent_kernel_read_readvariableop8
4savev2_lstm_12_lstm_cell_24_bias_read_readvariableop:
6savev2_lstm_13_lstm_cell_25_kernel_read_readvariableopD
@savev2_lstm_13_lstm_cell_25_recurrent_kernel_read_readvariableop8
4savev2_lstm_13_lstm_cell_25_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop&
"savev2_total_2_read_readvariableop&
"savev2_count_2_read_readvariableop5
1savev2_adam_dense_22_kernel_m_read_readvariableop3
/savev2_adam_dense_22_bias_m_read_readvariableop5
1savev2_adam_dense_23_kernel_m_read_readvariableop3
/savev2_adam_dense_23_bias_m_read_readvariableopA
=savev2_adam_lstm_12_lstm_cell_24_kernel_m_read_readvariableopK
Gsavev2_adam_lstm_12_lstm_cell_24_recurrent_kernel_m_read_readvariableop?
;savev2_adam_lstm_12_lstm_cell_24_bias_m_read_readvariableopA
=savev2_adam_lstm_13_lstm_cell_25_kernel_m_read_readvariableopK
Gsavev2_adam_lstm_13_lstm_cell_25_recurrent_kernel_m_read_readvariableop?
;savev2_adam_lstm_13_lstm_cell_25_bias_m_read_readvariableop5
1savev2_adam_dense_22_kernel_v_read_readvariableop3
/savev2_adam_dense_22_bias_v_read_readvariableop5
1savev2_adam_dense_23_kernel_v_read_readvariableop3
/savev2_adam_dense_23_bias_v_read_readvariableopA
=savev2_adam_lstm_12_lstm_cell_24_kernel_v_read_readvariableopK
Gsavev2_adam_lstm_12_lstm_cell_24_recurrent_kernel_v_read_readvariableop?
;savev2_adam_lstm_12_lstm_cell_24_bias_v_read_readvariableopA
=savev2_adam_lstm_13_lstm_cell_25_kernel_v_read_readvariableopK
Gsavev2_adam_lstm_13_lstm_cell_25_recurrent_kernel_v_read_readvariableop?
;savev2_adam_lstm_13_lstm_cell_25_bias_v_read_readvariableop
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
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:**
dtype0*?
value?B?*B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:**
dtype0*g
value^B\*B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_22_kernel_read_readvariableop(savev2_dense_22_bias_read_readvariableop*savev2_dense_23_kernel_read_readvariableop(savev2_dense_23_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop6savev2_lstm_12_lstm_cell_24_kernel_read_readvariableop@savev2_lstm_12_lstm_cell_24_recurrent_kernel_read_readvariableop4savev2_lstm_12_lstm_cell_24_bias_read_readvariableop6savev2_lstm_13_lstm_cell_25_kernel_read_readvariableop@savev2_lstm_13_lstm_cell_25_recurrent_kernel_read_readvariableop4savev2_lstm_13_lstm_cell_25_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop"savev2_total_2_read_readvariableop"savev2_count_2_read_readvariableop1savev2_adam_dense_22_kernel_m_read_readvariableop/savev2_adam_dense_22_bias_m_read_readvariableop1savev2_adam_dense_23_kernel_m_read_readvariableop/savev2_adam_dense_23_bias_m_read_readvariableop=savev2_adam_lstm_12_lstm_cell_24_kernel_m_read_readvariableopGsavev2_adam_lstm_12_lstm_cell_24_recurrent_kernel_m_read_readvariableop;savev2_adam_lstm_12_lstm_cell_24_bias_m_read_readvariableop=savev2_adam_lstm_13_lstm_cell_25_kernel_m_read_readvariableopGsavev2_adam_lstm_13_lstm_cell_25_recurrent_kernel_m_read_readvariableop;savev2_adam_lstm_13_lstm_cell_25_bias_m_read_readvariableop1savev2_adam_dense_22_kernel_v_read_readvariableop/savev2_adam_dense_22_bias_v_read_readvariableop1savev2_adam_dense_23_kernel_v_read_readvariableop/savev2_adam_dense_23_bias_v_read_readvariableop=savev2_adam_lstm_12_lstm_cell_24_kernel_v_read_readvariableopGsavev2_adam_lstm_12_lstm_cell_24_recurrent_kernel_v_read_readvariableop;savev2_adam_lstm_12_lstm_cell_24_bias_v_read_readvariableop=savev2_adam_lstm_13_lstm_cell_25_kernel_v_read_readvariableopGsavev2_adam_lstm_13_lstm_cell_25_recurrent_kernel_v_read_readvariableop;savev2_adam_lstm_13_lstm_cell_25_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *8
dtypes.
,2*	2
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
_input_shapes?
?: :	?::20:0: : : : : :	?:
??:?:	?:	2?:?: : : : : : :	?::20:0:	?:
??:?:	?:	2?:?:	?::20:0:	?:
??:?:	?:	2?:?: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	?: 

_output_shapes
::$ 

_output_shapes

:20: 

_output_shapes
:0:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :%
!

_output_shapes
:	?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:%!

_output_shapes
:	?:%!

_output_shapes
:	2?:!

_output_shapes	
:?:
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
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	?: 

_output_shapes
::$ 

_output_shapes

:20: 

_output_shapes
:0:%!

_output_shapes
:	?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:%!

_output_shapes
:	?:%!

_output_shapes
:	2?:!

_output_shapes	
:?:% !

_output_shapes
:	?: !

_output_shapes
::$" 

_output_shapes

:20: #

_output_shapes
:0:%$!

_output_shapes
:	?:&%"
 
_output_shapes
:
??:!&

_output_shapes	
:?:%'!

_output_shapes
:	?:%(!

_output_shapes
:	2?:!)

_output_shapes	
:?:*

_output_shapes
: 
? 
?
C__inference_lstm_12_layer_call_and_return_conditional_losses_157389

inputs#
mlclstm_readvariableop_resource%
!mlclstm_readvariableop_1_resource%
!mlclstm_readvariableop_2_resource
identity??MLCLSTM/ReadVariableOp?MLCLSTM/ReadVariableOp_1?MLCLSTM/ReadVariableOp_2D
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
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????2	
zeros_1m
MLCLSTM/hidden_sizeConst*
_output_shapes
: *
dtype0*
value
B :?2
MLCLSTM/hidden_sizem
MLCLSTM/output_sizeConst*
_output_shapes
: *
dtype0*
value
B :?2
MLCLSTM/output_size?
MLCLSTM/ReadVariableOpReadVariableOpmlclstm_readvariableop_resource*
_output_shapes
:	?*
dtype02
MLCLSTM/ReadVariableOp?
MLCLSTM/ReadVariableOp_1ReadVariableOp!mlclstm_readvariableop_1_resource* 
_output_shapes
:
??*
dtype02
MLCLSTM/ReadVariableOp_1?
MLCLSTM/ReadVariableOp_2ReadVariableOp!mlclstm_readvariableop_2_resource*
_output_shapes	
:?*
dtype02
MLCLSTM/ReadVariableOp_2?
MLCLSTMMLCLSTMinputsMLCLSTM/hidden_size:output:0MLCLSTM/output_size:output:0MLCLSTM/ReadVariableOp:value:0 MLCLSTM/ReadVariableOp_1:value:0 MLCLSTM/ReadVariableOp_2:value:0*
T0*,
_output_shapes
:?????????`?*
dropout%    *
return_sequences(2	
MLCLSTM?
IdentityIdentityMLCLSTM:output:0^MLCLSTM/ReadVariableOp^MLCLSTM/ReadVariableOp_1^MLCLSTM/ReadVariableOp_2*
T0*,
_output_shapes
:?????????`?2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????`:::20
MLCLSTM/ReadVariableOpMLCLSTM/ReadVariableOp24
MLCLSTM/ReadVariableOp_1MLCLSTM/ReadVariableOp_124
MLCLSTM/ReadVariableOp_2MLCLSTM/ReadVariableOp_2:S O
+
_output_shapes
:?????????`
 
_user_specified_nameinputs
?

?
D__inference_dense_22_layer_call_and_return_conditional_losses_158180

inputs%
!mlcmatmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MLCMatMul/ReadVariableOp?
MLCMatMul/ReadVariableOpReadVariableOp!mlcmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MLCMatMul/ReadVariableOp?
	MLCMatMul	MLCMatMulinputs MLCMatMul/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????`*

input_rank2
	MLCMatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMLCMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????`2	
BiasAdd\
TanhTanhBiasAdd:output:0*
T0*+
_output_shapes
:?????????`2
Tanh?
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MLCMatMul/ReadVariableOp*
T0*+
_output_shapes
:?????????`2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :?????????`?::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
MLCMatMul/ReadVariableOpMLCMatMul/ReadVariableOp:T P
,
_output_shapes
:?????????`?
 
_user_specified_nameinputs
?"
?
C__inference_lstm_13_layer_call_and_return_conditional_losses_157296

inputs#
mlclstm_readvariableop_resource%
!mlclstm_readvariableop_1_resource%
!mlclstm_readvariableop_2_resource
identity??MLCLSTM/ReadVariableOp?MLCLSTM/ReadVariableOp_1?MLCLSTM/ReadVariableOp_2D
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
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????22	
zeros_1l
MLCLSTM/hidden_sizeConst*
_output_shapes
: *
dtype0*
value	B :22
MLCLSTM/hidden_sizel
MLCLSTM/output_sizeConst*
_output_shapes
: *
dtype0*
value	B :22
MLCLSTM/output_size?
MLCLSTM/ReadVariableOpReadVariableOpmlclstm_readvariableop_resource*
_output_shapes
:	?*
dtype02
MLCLSTM/ReadVariableOp?
MLCLSTM/ReadVariableOp_1ReadVariableOp!mlclstm_readvariableop_1_resource*
_output_shapes
:	2?*
dtype02
MLCLSTM/ReadVariableOp_1?
MLCLSTM/ReadVariableOp_2ReadVariableOp!mlclstm_readvariableop_2_resource*
_output_shapes	
:?*
dtype02
MLCLSTM/ReadVariableOp_2?
MLCLSTMMLCLSTMinputsMLCLSTM/hidden_size:output:0MLCLSTM/output_size:output:0MLCLSTM/ReadVariableOp:value:0 MLCLSTM/ReadVariableOp_1:value:0 MLCLSTM/ReadVariableOp_2:value:0*
T0*+
_output_shapes
:?????????2*
dropout%    2	
MLCLSTMo
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   2
Reshape/shapey
ReshapeReshapeMLCLSTM:output:0Reshape/shape:output:0*
T0*'
_output_shapes
:?????????22	
Reshape?
IdentityIdentityReshape:output:0^MLCLSTM/ReadVariableOp^MLCLSTM/ReadVariableOp_1^MLCLSTM/ReadVariableOp_2*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????:::20
MLCLSTM/ReadVariableOpMLCLSTM/ReadVariableOp24
MLCLSTM/ReadVariableOp_1MLCLSTM/ReadVariableOp_124
MLCLSTM/ReadVariableOp_2MLCLSTM/ReadVariableOp_2:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
K
lstm_12_input:
serving_default_lstm_12_input:0?????????`<
dense_230
StatefulPartitionedCall:0?????????0tensorflow/serving/predict:??
?8
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
	optimizer
regularization_losses
	variables
trainable_variables
		keras_api


signatures
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses"?5
_tf_keras_sequential?5{"class_name": "Sequential", "name": "sequential_19", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_19", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 96, 5]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "lstm_12_input"}}, {"class_name": "LSTM", "config": {"name": "lstm_12", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 96, 5]}, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 150, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}}, {"class_name": "Dense", "config": {"name": "dense_22", "trainable": true, "dtype": "float32", "units": 20, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "LSTM", "config": {"name": "lstm_13", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 50, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}}, {"class_name": "Dense", "config": {"name": "dense_23", "trainable": true, "dtype": "float32", "units": 48, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 5]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 96, 5]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_19", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 96, 5]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "lstm_12_input"}}, {"class_name": "LSTM", "config": {"name": "lstm_12", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 96, 5]}, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 150, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}}, {"class_name": "Dense", "config": {"name": "dense_22", "trainable": true, "dtype": "float32", "units": 20, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "LSTM", "config": {"name": "lstm_13", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 50, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}}, {"class_name": "Dense", "config": {"name": "dense_23", "trainable": true, "dtype": "float32", "units": 48, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": "mean_squared_error", "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "mae", "dtype": "float32", "fn": "mean_absolute_error"}}, {"class_name": "MeanMetricWrapper", "config": {"name": "acc", "dtype": "float32", "fn": "categorical_accuracy"}}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?
cell

state_spec
regularization_losses
	variables
trainable_variables
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_rnn_layer?
{"class_name": "LSTM", "name": "lstm_12", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 96, 5]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "lstm_12", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 96, 5]}, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 150, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 5]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 96, 5]}}
?

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_22", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_22", "trainable": true, "dtype": "float32", "units": 20, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 150}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 96, 150]}}
?
cell

state_spec
regularization_losses
	variables
trainable_variables
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?

_tf_keras_rnn_layer?	{"class_name": "LSTM", "name": "lstm_13", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "lstm_13", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 50, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 20]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 96, 20]}}
?

kernel
bias
regularization_losses
 	variables
!trainable_variables
"	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_23", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_23", "trainable": true, "dtype": "float32", "units": 48, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 50}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50]}}
?
#iter

$beta_1

%beta_2
	&decay
'learning_ratemlmmmnmo(mp)mq*mr+ms,mt-muvvvwvxvy(vz)v{*v|+v},v~-v"
	optimizer
 "
trackable_list_wrapper
f
(0
)1
*2
3
4
+5
,6
-7
8
9"
trackable_list_wrapper
f
(0
)1
*2
3
4
+5
,6
-7
8
9"
trackable_list_wrapper
?
regularization_losses
.layer_regularization_losses
/layer_metrics
	variables
0non_trainable_variables
trainable_variables
1metrics

2layers
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
?

(kernel
)recurrent_kernel
*bias
3regularization_losses
4	variables
5trainable_variables
6	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "LSTMCell", "name": "lstm_cell_24", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "lstm_cell_24", "trainable": true, "dtype": "float32", "units": 150, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
(0
)1
*2"
trackable_list_wrapper
5
(0
)1
*2"
trackable_list_wrapper
?
regularization_losses
7layer_regularization_losses
8layer_metrics
	variables
9non_trainable_variables
trainable_variables
:metrics

;layers

<states
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
": 	?2dense_22/kernel
:2dense_22/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
regularization_losses
=layer_regularization_losses
>layer_metrics
	variables
?non_trainable_variables
trainable_variables
@metrics

Alayers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?

+kernel
,recurrent_kernel
-bias
Bregularization_losses
C	variables
Dtrainable_variables
E	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "LSTMCell", "name": "lstm_cell_25", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "lstm_cell_25", "trainable": true, "dtype": "float32", "units": 50, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
+0
,1
-2"
trackable_list_wrapper
5
+0
,1
-2"
trackable_list_wrapper
?
regularization_losses
Flayer_regularization_losses
Glayer_metrics
	variables
Hnon_trainable_variables
trainable_variables
Imetrics

Jlayers

Kstates
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
!:202dense_23/kernel
:02dense_23/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
regularization_losses
Llayer_regularization_losses
Mlayer_metrics
 	variables
Nnon_trainable_variables
!trainable_variables
Ometrics

Players
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
.:,	?2lstm_12/lstm_cell_24/kernel
9:7
??2%lstm_12/lstm_cell_24/recurrent_kernel
(:&?2lstm_12/lstm_cell_24/bias
.:,	?2lstm_13/lstm_cell_25/kernel
8:6	2?2%lstm_13/lstm_cell_25/recurrent_kernel
(:&?2lstm_13/lstm_cell_25/bias
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
5
Q0
R1
S2"
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
5
(0
)1
*2"
trackable_list_wrapper
5
(0
)1
*2"
trackable_list_wrapper
?
3regularization_losses
Tlayer_regularization_losses
Ulayer_metrics
4	variables
Vnon_trainable_variables
5trainable_variables
Wmetrics

Xlayers
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
0"
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
+0
,1
-2"
trackable_list_wrapper
5
+0
,1
-2"
trackable_list_wrapper
?
Bregularization_losses
Ylayer_regularization_losses
Zlayer_metrics
C	variables
[non_trainable_variables
Dtrainable_variables
\metrics

]layers
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
0"
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
	^total
	_count
`	variables
a	keras_api"?
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
?
	btotal
	ccount
d
_fn_kwargs
e	variables
f	keras_api"?
_tf_keras_metric?{"class_name": "MeanMetricWrapper", "name": "mae", "dtype": "float32", "config": {"name": "mae", "dtype": "float32", "fn": "mean_absolute_error"}}
?
	gtotal
	hcount
i
_fn_kwargs
j	variables
k	keras_api"?
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
^0
_1"
trackable_list_wrapper
-
`	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
b0
c1"
trackable_list_wrapper
-
e	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
g0
h1"
trackable_list_wrapper
-
j	variables"
_generic_user_object
':%	?2Adam/dense_22/kernel/m
 :2Adam/dense_22/bias/m
&:$202Adam/dense_23/kernel/m
 :02Adam/dense_23/bias/m
3:1	?2"Adam/lstm_12/lstm_cell_24/kernel/m
>:<
??2,Adam/lstm_12/lstm_cell_24/recurrent_kernel/m
-:+?2 Adam/lstm_12/lstm_cell_24/bias/m
3:1	?2"Adam/lstm_13/lstm_cell_25/kernel/m
=:;	2?2,Adam/lstm_13/lstm_cell_25/recurrent_kernel/m
-:+?2 Adam/lstm_13/lstm_cell_25/bias/m
':%	?2Adam/dense_22/kernel/v
 :2Adam/dense_22/bias/v
&:$202Adam/dense_23/kernel/v
 :02Adam/dense_23/bias/v
3:1	?2"Adam/lstm_12/lstm_cell_24/kernel/v
>:<
??2,Adam/lstm_12/lstm_cell_24/recurrent_kernel/v
-:+?2 Adam/lstm_12/lstm_cell_24/bias/v
3:1	?2"Adam/lstm_13/lstm_cell_25/kernel/v
=:;	2?2,Adam/lstm_13/lstm_cell_25/recurrent_kernel/v
-:+?2 Adam/lstm_13/lstm_cell_25/bias/v
?2?
.__inference_sequential_19_layer_call_fn_157989
.__inference_sequential_19_layer_call_fn_157964
.__inference_sequential_19_layer_call_fn_157744
.__inference_sequential_19_layer_call_fn_157691?
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
!__inference__wrapped_model_157024?
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
annotations? *0?-
+?(
lstm_12_input?????????`
?2?
I__inference_sequential_19_layer_call_and_return_conditional_losses_157609
I__inference_sequential_19_layer_call_and_return_conditional_losses_157859
I__inference_sequential_19_layer_call_and_return_conditional_losses_157939
I__inference_sequential_19_layer_call_and_return_conditional_losses_157637?
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
(__inference_lstm_12_layer_call_fn_158169
(__inference_lstm_12_layer_call_fn_158158
(__inference_lstm_12_layer_call_fn_158068
(__inference_lstm_12_layer_call_fn_158079?
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
C__inference_lstm_12_layer_call_and_return_conditional_losses_158023
C__inference_lstm_12_layer_call_and_return_conditional_losses_158147
C__inference_lstm_12_layer_call_and_return_conditional_losses_158057
C__inference_lstm_12_layer_call_and_return_conditional_losses_158113?
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
)__inference_dense_22_layer_call_fn_158189?
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
D__inference_dense_22_layer_call_and_return_conditional_losses_158180?
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
?2?
(__inference_lstm_13_layer_call_fn_158272
(__inference_lstm_13_layer_call_fn_158377
(__inference_lstm_13_layer_call_fn_158366
(__inference_lstm_13_layer_call_fn_158283?
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
C__inference_lstm_13_layer_call_and_return_conditional_losses_158225
C__inference_lstm_13_layer_call_and_return_conditional_losses_158319
C__inference_lstm_13_layer_call_and_return_conditional_losses_158261
C__inference_lstm_13_layer_call_and_return_conditional_losses_158355?
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
)__inference_dense_23_layer_call_fn_158397?
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
D__inference_dense_23_layer_call_and_return_conditional_losses_158388?
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
$__inference_signature_wrapper_157779lstm_12_input"?
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
!__inference__wrapped_model_157024}
()*+,-:?7
0?-
+?(
lstm_12_input?????????`
? "3?0
.
dense_23"?
dense_23?????????0?
D__inference_dense_22_layer_call_and_return_conditional_losses_158180e4?1
*?'
%?"
inputs?????????`?
? ")?&
?
0?????????`
? ?
)__inference_dense_22_layer_call_fn_158189X4?1
*?'
%?"
inputs?????????`?
? "??????????`?
D__inference_dense_23_layer_call_and_return_conditional_losses_158388\/?,
%?"
 ?
inputs?????????2
? "%?"
?
0?????????0
? |
)__inference_dense_23_layer_call_fn_158397O/?,
%?"
 ?
inputs?????????2
? "??????????0?
C__inference_lstm_12_layer_call_and_return_conditional_losses_158023?()*O?L
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
C__inference_lstm_12_layer_call_and_return_conditional_losses_158057?()*O?L
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
C__inference_lstm_12_layer_call_and_return_conditional_losses_158113r()*??<
5?2
$?!
inputs?????????`

 
p

 
? "*?'
 ?
0?????????`?
? ?
C__inference_lstm_12_layer_call_and_return_conditional_losses_158147r()*??<
5?2
$?!
inputs?????????`

 
p 

 
? "*?'
 ?
0?????????`?
? ?
(__inference_lstm_12_layer_call_fn_158068~()*O?L
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
(__inference_lstm_12_layer_call_fn_158079~()*O?L
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
(__inference_lstm_12_layer_call_fn_158158e()*??<
5?2
$?!
inputs?????????`

 
p

 
? "??????????`??
(__inference_lstm_12_layer_call_fn_158169e()*??<
5?2
$?!
inputs?????????`

 
p 

 
? "??????????`??
C__inference_lstm_13_layer_call_and_return_conditional_losses_158225}+,-O?L
E?B
4?1
/?,
inputs/0??????????????????

 
p

 
? "%?"
?
0?????????2
? ?
C__inference_lstm_13_layer_call_and_return_conditional_losses_158261}+,-O?L
E?B
4?1
/?,
inputs/0??????????????????

 
p 

 
? "%?"
?
0?????????2
? ?
C__inference_lstm_13_layer_call_and_return_conditional_losses_158319m+,-??<
5?2
$?!
inputs?????????`

 
p

 
? "%?"
?
0?????????2
? ?
C__inference_lstm_13_layer_call_and_return_conditional_losses_158355m+,-??<
5?2
$?!
inputs?????????`

 
p 

 
? "%?"
?
0?????????2
? ?
(__inference_lstm_13_layer_call_fn_158272p+,-O?L
E?B
4?1
/?,
inputs/0??????????????????

 
p

 
? "??????????2?
(__inference_lstm_13_layer_call_fn_158283p+,-O?L
E?B
4?1
/?,
inputs/0??????????????????

 
p 

 
? "??????????2?
(__inference_lstm_13_layer_call_fn_158366`+,-??<
5?2
$?!
inputs?????????`

 
p

 
? "??????????2?
(__inference_lstm_13_layer_call_fn_158377`+,-??<
5?2
$?!
inputs?????????`

 
p 

 
? "??????????2?
I__inference_sequential_19_layer_call_and_return_conditional_losses_157609w
()*+,-B??
8?5
+?(
lstm_12_input?????????`
p

 
? "%?"
?
0?????????0
? ?
I__inference_sequential_19_layer_call_and_return_conditional_losses_157637w
()*+,-B??
8?5
+?(
lstm_12_input?????????`
p 

 
? "%?"
?
0?????????0
? ?
I__inference_sequential_19_layer_call_and_return_conditional_losses_157859p
()*+,-;?8
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
I__inference_sequential_19_layer_call_and_return_conditional_losses_157939p
()*+,-;?8
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
.__inference_sequential_19_layer_call_fn_157691j
()*+,-B??
8?5
+?(
lstm_12_input?????????`
p

 
? "??????????0?
.__inference_sequential_19_layer_call_fn_157744j
()*+,-B??
8?5
+?(
lstm_12_input?????????`
p 

 
? "??????????0?
.__inference_sequential_19_layer_call_fn_157964c
()*+,-;?8
1?.
$?!
inputs?????????`
p

 
? "??????????0?
.__inference_sequential_19_layer_call_fn_157989c
()*+,-;?8
1?.
$?!
inputs?????????`
p 

 
? "??????????0?
$__inference_signature_wrapper_157779?
()*+,-K?H
? 
A?>
<
lstm_12_input+?(
lstm_12_input?????????`"3?0
.
dense_23"?
dense_23?????????0